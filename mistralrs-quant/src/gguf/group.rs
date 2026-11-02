use std::sync::Arc;

use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Result, Tensor,
};

use crate::{ActivationCalibration, GroupEntry, GroupKey, GroupKind, GroupMember};

use super::GgufMatMul;

const GGUF_GROUP_FORMAT: &str = "gguf";

impl GgufMatMul {
    pub(super) fn gguf_group_member(
        &self,
        xs: &Tensor,
        kind: GroupKind<'_>,
    ) -> Result<Option<GroupMember>> {
        if matches!(kind, GroupKind::Gather { .. }) || self.b.is_some() {
            return Ok(None);
        }
        let QMatMul::QTensor(weight) = &self.w else {
            return Ok(None);
        };
        let Ok((rows, columns)) = weight.shape().dims2() else {
            return Ok(None);
        };
        // A member with a prepared packed-affine kernel is faster on its own forward.
        #[cfg(all(feature = "cuda", has_marlin_kernels))]
        if xs.device().is_cuda() && xs.rank() > 0 {
            let tokens = xs.elem_count() / xs.dim(candle_core::D::Minus1)?.max(1);
            if tokens >= super::GGUF_AFFINE_MIN_BATCH
                && self
                    .packed_affine_for(tokens, xs.dtype(), xs.device())?
                    .is_some()
            {
                return Ok(None);
            }
        }
        Ok(Some(GroupMember {
            key: GroupKey {
                format: GGUF_GROUP_FORMAT,
                dtype: xs.dtype(),
                shape: [1, rows, columns],
                calibration: ActivationCalibration::Dynamic,
            },
            forward: forward_group,
            operand: Arc::clone(weight) as Arc<dyn std::any::Any + Send + Sync>,
        }))
    }
}

fn weight<'a>(entry: &'a GroupEntry<'_>) -> Option<&'a QTensor> {
    entry.operand::<QTensor>(GGUF_GROUP_FORMAT)
}

fn weights<'a>(entries: &'a [GroupEntry<'_>]) -> Option<Vec<&'a QTensor>> {
    entries.iter().map(weight).collect()
}

fn forward_group(
    xs: &Tensor,
    entries: &[GroupEntry<'_>],
    kind: GroupKind<'_>,
) -> Result<Option<Vec<Tensor>>> {
    if entries.iter().any(|entry| entry.method.has_bias()) {
        return Ok(None);
    }
    if xs.device().is_cpu() {
        return forward_group_cpu(xs, entries, kind);
    }
    #[cfg(feature = "cuda")]
    if xs.device().is_cuda() {
        return cuda::forward_group(xs, entries, kind);
    }
    Ok(None)
}

/// aarch64 repacked GEMV: one input quantization and one parallel region for all m == 1 projections.
fn forward_group_cpu(
    xs: &Tensor,
    entries: &[GroupEntry<'_>],
    kind: GroupKind<'_>,
) -> Result<Option<Vec<Tensor>>> {
    if !matches!(kind, GroupKind::Dense) || xs.dtype() != DType::F32 {
        return Ok(None);
    }
    let Some(weights) = weights(entries) else {
        return Ok(None);
    };
    QTensor::gemv_fused_shared_lhs(&weights, xs)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::{quantized::QTensor, Result, Tensor};

    use super::{weight, weights};
    use crate::{
        gguf::{fast_mmq, fast_mmvq, GGUF_AFFINE_MIN_BATCH},
        GluActivationType, GroupEntry, GroupKind,
    };

    pub(super) fn forward_group(
        xs: &Tensor,
        entries: &[GroupEntry<'_>],
        kind: GroupKind<'_>,
    ) -> Result<Option<Vec<Tensor>>> {
        let Some((&columns, batch_dims)) = xs.dims().split_last() else {
            return Ok(None);
        };
        let tokens = batch_dims.iter().product::<usize>();
        if tokens == 0 {
            return Ok(None);
        }
        match kind {
            GroupKind::Dense => dense(xs, entries, columns, tokens),
            GroupKind::Glu { activation } => {
                let Some([gate, up]) = weights(entries).and_then(|w| <[_; 2]>::try_from(w).ok())
                else {
                    return Ok(None);
                };
                glu(xs, gate, up, columns, tokens, activation)
            }
            GroupKind::Ffn { activation } => {
                let Some([gate, up, down]) =
                    weights(entries).and_then(|w| <[_; 3]>::try_from(w).ok())
                else {
                    return Ok(None);
                };
                ffn(xs, [gate, up, down], columns, tokens, activation)
            }
            GroupKind::Gather { .. } => Ok(None),
        }
    }

    fn same_dtype(weights: &[&QTensor]) -> bool {
        weights.iter().all(|w| w.dtype() == weights[0].dtype())
    }

    fn dense(
        xs: &Tensor,
        entries: &[GroupEntry<'_>],
        columns: usize,
        tokens: usize,
    ) -> Result<Option<Vec<Tensor>>> {
        if let [left, right] = entries {
            if tokens == 1 {
                match (weight(left), weight(right)) {
                    (Some(quant), None) => {
                        if let Some(dense) = right.method.unquant_forward_weight() {
                            return fast_mmvq::try_dense_pair(quant, &dense, xs)
                                .map(|pair| pair.map(|(quant, dense)| vec![quant, dense]));
                        }
                    }
                    (None, Some(quant)) => {
                        if let Some(dense) = left.method.unquant_forward_weight() {
                            return fast_mmvq::try_dense_pair(quant, &dense, xs)
                                .map(|pair| pair.map(|(quant, dense)| vec![dense, quant]));
                        }
                    }
                    _ => {}
                }
            }
        }
        let Some(weights) = weights(entries) else {
            return Ok(None);
        };
        if weights
            .iter()
            .any(|w| w.shape().dims2().map_or(true, |(_, cols)| cols != columns))
        {
            return Ok(None);
        }
        if tokens <= fast_mmvq::MMVQ_MAX_BATCH {
            if !weights.iter().all(|w| fast_mmvq::supports(w.dtype())) {
                return Ok(None);
            }
            return match weights.as_slice() {
                [q, k, v] if same_dtype(&weights) => {
                    let (q, k, v) = fast_mmvq::fused_qkv(q, k, v, xs)?;
                    Ok(Some(vec![q, k, v]))
                }
                [left, right]
                    if same_dtype(&weights)
                        && fast_mmvq::prefers_fused_pair(
                            xs.device().as_cuda_device()?,
                            tokens,
                            [left.shape().dims2()?.0, right.shape().dims2()?.0],
                        ) =>
                {
                    let (left, right) = fast_mmvq::fused_pair(left, right, xs)?;
                    Ok(Some(vec![left, right]))
                }
                _ => fast_mmvq::shared_lhs(&weights, xs).map(Some),
            };
        }
        if !same_dtype(&weights) || !fast_mmq::supports(weights[0].dtype()) {
            return Ok(None);
        }
        fast_mmq::shared_lhs(&weights, xs).map(Some)
    }

    fn glu(
        xs: &Tensor,
        gate: &QTensor,
        up: &QTensor,
        columns: usize,
        tokens: usize,
        activation: GluActivationType,
    ) -> Result<Option<Vec<Tensor>>> {
        if gate.shape() != up.shape()
            || gate.dtype() != up.dtype()
            || gate.shape().dims2()?.1 != columns
        {
            return Ok(None);
        }
        let output = if tokens <= fast_mmvq::MMVQ_MAX_BATCH {
            if !fast_mmvq::supports_fused_glu(xs.dtype(), gate.dtype()) {
                return Ok(None);
            }
            fast_mmvq::fused_glu(gate, up, xs, activation)?
        } else {
            if !fast_mmq::supports(gate.dtype()) {
                return Ok(None);
            }
            fast_mmq::fused_glu(gate, up, xs, activation)?
        };
        Ok(Some(vec![output]))
    }

    fn ffn(
        xs: &Tensor,
        [gate, up, down]: [&QTensor; 3],
        columns: usize,
        tokens: usize,
        activation: GluActivationType,
    ) -> Result<Option<Vec<Tensor>>> {
        // Decode keeps gate/up fused and down separate; only prefill fuses all three.
        if !matches!(xs.rank(), 2 | 3)
            || tokens < GGUF_AFFINE_MIN_BATCH
            || tokens <= fast_mmvq::MMVQ_MAX_BATCH
        {
            return Ok(None);
        }
        let (intermediate, gate_columns) = gate.shape().dims2()?;
        let (_, down_columns) = down.shape().dims2()?;
        if gate.shape() != up.shape()
            || gate_columns != columns
            || down_columns != intermediate
            || gate.dtype() != up.dtype()
            || !fast_mmq::supports(gate.dtype())
            || !fast_mmq::supports(down.dtype())
            || !gate.device().same_device(xs.device())
            || !up.device().same_device(xs.device())
            || !down.device().same_device(xs.device())
        {
            return Ok(None);
        }
        Ok(Some(vec![fast_mmq::fused_ffn(
            gate, up, down, xs, activation,
        )?]))
    }
}
