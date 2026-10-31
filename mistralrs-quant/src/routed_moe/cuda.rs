use candle_core::{DType, Device, Result, Tensor, D};

use super::{Execution, RoutedExperts, GROUPED_PREFILL_MIN_TOKENS};
use crate::{GluActivationType, LoraExpertInputMode, LoraExpertProjection};

const GROUPED_MMQ_MIN_REUSE: f64 = 1.25;
const MMQ_TILE_ROWS: usize = 128;

fn contiguous_ids(ids: &Tensor) -> Result<Tensor> {
    let ids = ids.flatten_all()?.contiguous()?;
    if ids.layout().start_offset() == 0 {
        Ok(ids)
    } else {
        ids.force_contiguous()
    }
}

#[derive(Clone, Copy)]
pub(super) enum CudaRoute {
    PerAssignment,
    Grouped,
}

enum GroupedGateUp {
    Packed(Tensor),
    SortedPair { gate: Tensor, up: Tensor },
}

/// Routed GGUF expert batch as seen by the grouped-MMQ vs per-assignment GEMV cost rule.
#[derive(Clone, Copy, Debug)]
struct GroupedMmqShape {
    pub tokens: usize,
    pub experts: usize,
    pub topk: usize,
    pub gate_up_rows: usize,
    pub expert_bytes: usize,
}

fn count_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

impl GroupedMmqShape {
    fn expected_touched_experts(self) -> f64 {
        let experts = count_f64(self.experts);
        experts * (1.0 - (1.0 - 1.0 / experts).powf(count_f64(self.tokens * self.topk)))
    }

    // GEMV re-streams an expert per route, grouped once per touched expert: wins if re-reads miss L2 and tiles fill SMs.
    fn reuses_weights(self, device: crate::cuda_device::CudaDeviceInfo) -> bool {
        let touched = self.expected_touched_experts();
        count_f64(self.tokens * self.topk) >= GROUPED_MMQ_MIN_REUSE * touched
            && touched * count_f64(self.expert_bytes) > count_f64(device.l2_bytes)
            && touched * count_f64(self.gate_up_rows.div_ceil(MMQ_TILE_ROWS))
                >= count_f64(device.sms)
    }
}

impl RoutedExperts<'_> {
    pub(super) fn grouped_lora_preserves_route_order(&self) -> bool {
        self.gate
            .get_qtensor()
            .zip(self.up.get_qtensor())
            .is_some_and(|(gate, up)| {
                gate.dtype() == up.dtype() && crate::supports_mmq(gate.dtype())
            })
    }

    pub(super) fn select_cuda_fast_path(&self, forward: &Execution) -> Option<CudaRoute> {
        let Device::Cuda(dev) = forward.xs_flat.device() else {
            return None;
        };
        let grouped = forward.shape.num_tokens >= GROUPED_PREFILL_MIN_TOKENS
            || self.grouped_mmq_shape(forward).is_some_and(|shape| {
                shape.reuses_weights(crate::cuda_device::cuda_device_info(dev))
            });
        Some(if grouped {
            CudaRoute::Grouped
        } else {
            CudaRoute::PerAssignment
        })
    }

    fn grouped_mmq_shape(&self, forward: &Execution) -> Option<GroupedMmqShape> {
        let weights = [
            self.gate.get_qtensor()?,
            self.up.get_qtensor()?,
            self.down.get_qtensor()?,
        ];
        if !weights
            .iter()
            .all(|weight| crate::supports_mmq(weight.dtype()))
        {
            return None;
        }
        let bytes: usize = weights
            .iter()
            .map(|weight| {
                weight.shape().elem_count() / weight.dtype().block_size()
                    * weight.dtype().type_size()
            })
            .sum();
        Some(GroupedMmqShape {
            tokens: forward.shape.num_tokens,
            experts: self.num_experts,
            topk: forward.topk,
            gate_up_rows: weights[0].shape().dims()[1] + weights[1].shape().dims()[1],
            expert_bytes: bytes / self.num_experts,
        })
    }

    pub(super) fn forward_cuda(&self, forward: &Execution) -> Result<Option<Tensor>> {
        match self.select_cuda_fast_path(forward) {
            Some(CudaRoute::PerAssignment) => self
                .forward_decode(forward)
                .map_err(|err| err.context("moe experts fast decode")),
            Some(CudaRoute::Grouped) => self
                .forward_grouped(forward)
                .map_err(|err| err.context("moe experts fast grouped")),
            None => Ok(None),
        }
    }

    pub(super) fn forward_decode_lora(&self, forward: &Execution) -> Result<Option<Tensor>> {
        if self.members().iter().any(|member| member.has_bias()) {
            return Ok(None);
        }
        let Some(gate_qt) = self.gate.get_qtensor() else {
            return Ok(None);
        };
        let Some(up_qt) = self.up.get_qtensor() else {
            return Ok(None);
        };
        let Some(down_qt) = self.down.get_qtensor() else {
            return Ok(None);
        };
        let dev = forward.xs_flat.device().as_cuda_device()?;
        let topk_ids_flat = contiguous_ids(forward.topk_ids)?;
        let (ids_storage, ids_layout) = topk_ids_flat.storage_and_layout();
        let candle_core::Storage::Cuda(ids_cuda) = &*ids_storage else {
            return Ok(None);
        };
        if ids_layout.start_offset() != 0 {
            return Ok(None);
        }
        let ids = ids_cuda.as_cuda_slice::<u32>()?;
        let weights = crate::IndexedMoeLoraWeights::new(&gate_qt, &up_qt, &down_qt);
        let routing = crate::IndexedMoeRouting::new(
            ids,
            forward.shape.num_tokens,
            forward.topk,
            self.num_experts,
            dev,
        );
        let Some(decode) = crate::IndexedMoeLoraDecode::new(weights, routing)? else {
            return Ok(None);
        };
        let Some(gate_up) = decode.gate_up(forward.xs_flat)? else {
            return Ok(None);
        };

        let lora = forward.lora.expect("LoRA path requires execution");
        let gate_up =
            lora.add_gate_up_delta_combined_owned(forward.xs_flat, gate_up, forward.topk_ids)?;
        let intermediate = gate_up.dim(D::Minus1)? / 2;
        let down_input = forward.activation.split(&gate_up, intermediate)?;
        let down_input_flat =
            down_input.reshape((forward.shape.num_tokens * forward.topk, intermediate))?;
        let Some(down_base) = decode.down(&down_input_flat)? else {
            candle_core::bail!("indexed MoE LoRA down projection rejected a supported dtype");
        };
        let down = lora.add_delta_owned(
            LoraExpertProjection::Down,
            &down_input,
            down_base,
            forward.topk_ids,
            None,
            LoraExpertInputMode::RoutedRows,
        )?;

        let down = down.reshape((
            forward.shape.num_tokens * forward.topk,
            forward.shape.hidden_dim,
        ))?;
        let output = crate::moe_weighted_reduce_flat_same_dtype(
            &down,
            forward.topk_weights,
            forward.shape.num_tokens,
            forward.topk,
            dev,
        )?;
        Ok(Some(output.to_dtype(forward.original_dtype)?))
    }

    fn forward_decode(&self, forward: &Execution) -> Result<Option<Tensor>> {
        if let Some(activation) = forward.activation.fused {
            if let Some(output) = crate::try_routed_ffn(
                forward.xs_flat,
                forward.topk_ids,
                forward.topk_weights,
                [self.gate, self.up, self.down],
                activation,
            )? {
                return Ok(Some(output.to_dtype(forward.original_dtype)?));
            }
        }

        if self.members().iter().any(|member| member.has_bias()) {
            return Ok(None);
        }
        let dev = forward.xs_flat.device().as_cuda_device()?;

        let gate_qt = match self.gate.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let up_qt = match self.up.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let down_qt = match self.down.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };

        let topk_ids_flat = contiguous_ids(forward.topk_ids)?;
        let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
        let ti_cuda = match &*ti_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
        assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

        let tw_f32 = forward
            .topk_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (tw_storage, tw_layout) = tw_f32.storage_and_layout();
        let tw_cuda = match &*tw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let (tw_ptr, _tw_guard) =
            crate::utils::slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &stream);
        let tw_ptr = tw_ptr as *const f32;

        let act_type = match forward.activation.fused {
            Some(GluActivationType::Gelu) => crate::ACT_GELU_PYTORCH_TANH,
            Some(GluActivationType::Silu) => crate::ACT_SILU,
            _ => return Ok(None),
        };

        let result = unsafe {
            crate::indexed_moe_fused_decode(
                &gate_qt,
                &up_qt,
                &down_qt,
                forward.xs_flat,
                ti_u32_slice,
                tw_ptr,
                forward.shape.num_tokens,
                forward.topk,
                act_type,
                dev,
            )?
        };

        Ok(Some(result.to_dtype(forward.original_dtype)?))
    }

    pub(super) fn forward_grouped(&self, forward: &Execution) -> Result<Option<Tensor>> {
        if self.members().iter().any(|member| member.has_bias()) {
            return Ok(None);
        }
        let topk = forward.topk;
        let num_experts = self.num_experts;
        let total_assignments = forward.shape.num_tokens * topk;

        let gate_qt = match self.gate.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let up_qt = match self.up.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };
        let down_qt = match self.down.get_qtensor() {
            Some(qt) => qt,
            None => return Ok(None),
        };

        let use_mmq_gate_up =
            gate_qt.dtype() == up_qt.dtype() && crate::supports_mmq(gate_qt.dtype());
        if forward.lora.is_some() && !use_mmq_gate_up {
            return Ok(None);
        }

        let dev = forward.xs_flat.device().as_cuda_device()?;

        let topk_ids_flat = contiguous_ids(forward.topk_ids)?;
        let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
        let ti_cuda = match &*ti_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
        assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

        let (expert_bounds, sorted_token_ids, sorted_source_ids) =
            crate::moe_dispatch_build(ti_u32_slice, total_assignments, num_experts, topk, dev)?;

        let gate_up = if use_mmq_gate_up {
            GroupedGateUp::Packed(crate::grouped_moe_mmq_pair_packed(
                &gate_qt,
                &up_qt,
                forward.xs_flat,
                &sorted_source_ids,
                &sorted_token_ids,
                &expert_bounds,
                total_assignments,
                topk,
                num_experts,
                dev,
            )?)
        } else {
            let (input_q8, k, k_padded) = crate::quantize_input_q8_1(forward.xs_flat, dev)?;

            let gate = crate::grouped_moe_gemm_prequantized(
                &gate_qt,
                &input_q8,
                k,
                k_padded,
                &expert_bounds,
                &sorted_token_ids,
                None,
                total_assignments,
                topk,
                num_experts,
                1,
                dev,
            )?;

            let up = crate::grouped_moe_gemm_prequantized(
                &up_qt,
                &input_q8,
                k,
                k_padded,
                &expert_bounds,
                &sorted_token_ids,
                None,
                total_assignments,
                topk,
                num_experts,
                1,
                dev,
            )?;

            drop(input_q8);
            GroupedGateUp::SortedPair { gate, up }
        };

        let lora_activated = if let Some(lora) = forward.lora {
            let GroupedGateUp::Packed(gate_up) = &gate_up else {
                return Ok(None);
            };
            let gate_up = lora.add_gate_up_delta_combined_owned(
                forward.xs_flat,
                gate_up.to_dtype(forward.original_dtype)?,
                forward.topk_ids,
            )?;
            let intermediate = gate_up.dim(D::Minus1)? / 2;
            Some(forward.activation.split(&gate_up, intermediate)?)
        } else {
            None
        };

        let tw_f32 = forward
            .topk_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (tw_storage, tw_layout) = tw_f32.storage_and_layout();
        let tw_cuda = match &*tw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => return Ok(None),
        };
        let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;
        let stream = dev.cuda_stream();
        let (tw_ptr, _tw_guard) =
            crate::utils::slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &stream);
        let tw_ptr = tw_ptr as *const f32;

        let glu_activation = forward
            .activation
            .fused
            .filter(|activation| !matches!(activation, GluActivationType::Sigmoid));

        let down = if let (true, Some(glu_activation)) =
            (crate::supports_mmq(down_qt.dtype()), glu_activation)
        {
            if let (Some(lora), Some(activated)) = (forward.lora, lora_activated.as_ref()) {
                let intermediate = activated.dim(D::Minus1)?;
                let activated_flat = activated.reshape((total_assignments, intermediate))?;
                let down_assignments = crate::grouped_moe_mmq(
                    &down_qt,
                    &activated_flat,
                    &sorted_token_ids,
                    &sorted_token_ids,
                    &expert_bounds,
                    total_assignments,
                    forward.shape.num_tokens,
                    num_experts,
                    dev,
                )?
                .to_dtype(activated.dtype())?
                .reshape((
                    forward.shape.num_tokens,
                    topk,
                    forward.shape.hidden_dim,
                ))?;
                let down = lora.add_delta_owned(
                    LoraExpertProjection::Down,
                    activated,
                    down_assignments,
                    forward.topk_ids,
                    None,
                    LoraExpertInputMode::RoutedRows,
                )?;
                crate::moe_weighted_reduce_flat_same_dtype(
                    &down.reshape((total_assignments, forward.shape.hidden_dim))?,
                    forward.topk_weights,
                    forward.shape.num_tokens,
                    topk,
                    dev,
                )?
            } else {
                let down_assignments = match &gate_up {
                    GroupedGateUp::Packed(gate_up) => crate::grouped_moe_mmq_from_glu_packed(
                        &down_qt,
                        gate_up,
                        &sorted_token_ids,
                        &sorted_token_ids,
                        &expert_bounds,
                        total_assignments,
                        forward.shape.num_tokens,
                        num_experts,
                        glu_activation as i32,
                        dev,
                    )?,
                    GroupedGateUp::SortedPair { gate, up } => {
                        crate::grouped_moe_mmq_from_glu_sorted_pair(
                            &down_qt,
                            gate,
                            up,
                            &sorted_token_ids,
                            &expert_bounds,
                            total_assignments,
                            forward.shape.num_tokens,
                            num_experts,
                            glu_activation as i32,
                            dev,
                        )?
                    }
                };
                if forward.original_dtype == DType::BF16 {
                    unsafe {
                        crate::moe_weighted_reduce_flat_bf16(
                            &down_assignments,
                            tw_ptr,
                            forward.shape.num_tokens,
                            topk,
                            dev,
                        )?
                    }
                } else {
                    unsafe {
                        crate::moe_weighted_reduce_flat(
                            &down_assignments,
                            tw_ptr,
                            forward.shape.num_tokens,
                            topk,
                            dev,
                        )?
                    }
                }
            }
        } else {
            let (activated, down_input_dim1) = match &lora_activated {
                Some(activated) => (
                    activated.reshape((total_assignments, activated.dim(D::Minus1)?))?,
                    2,
                ),
                None => match &gate_up {
                    GroupedGateUp::Packed(gate_up) => (
                        forward
                            .activation
                            .split(gate_up, gate_up.dim(D::Minus1)? / 2)?,
                        2,
                    ),
                    GroupedGateUp::SortedPair { gate, up } => {
                        (forward.activation.apply(gate, up)?, 0)
                    }
                },
            };

            let (down_input_q8, down_k, down_k_padded) =
                crate::quantize_input_q8_1(&activated, dev)?;

            let down = crate::grouped_moe_gemm_prequantized(
                &down_qt,
                &down_input_q8,
                down_k,
                down_k_padded,
                &expert_bounds,
                &sorted_token_ids,
                Some((tw_ptr, 0)),
                total_assignments,
                topk,
                num_experts,
                down_input_dim1,
                dev,
            )?;

            if let (Some(lora), Some(activated)) = (forward.lora, lora_activated.as_ref()) {
                let delta_base = Tensor::zeros(
                    (forward.shape.num_tokens, topk, forward.shape.hidden_dim),
                    activated.dtype(),
                    activated.device(),
                )?;
                let delta = lora.add_delta_owned(
                    LoraExpertProjection::Down,
                    activated,
                    delta_base,
                    forward.topk_ids,
                    Some(forward.topk_weights),
                    LoraExpertInputMode::RoutedRows,
                )?;
                (down.to_dtype(DType::F32)? + delta.to_dtype(DType::F32)?.sum(D::Minus2)?)?
            } else {
                down
            }
        };

        if down.dtype() == forward.original_dtype {
            Ok(Some(down))
        } else {
            Ok(Some(down.to_dtype(forward.original_dtype)?))
        }
    }
}
