use super::*;

pub enum DistributedKind {
    ColumnParallel,
    RowParallel,
    Replicated,
}

/// The fused operation requested from a group of layers that read the same input.
#[derive(Clone, Copy, Debug)]
pub enum GroupKind<'a> {
    /// One output per member.
    Dense,
    /// Expert stacks routed by `indices`; one output per member.
    Gather { indices: &'a Tensor },
    /// Members `[gate, up]`; one output `act(gate(x)) * up(x)`.
    Glu { activation: GluActivationType },
    /// Members `[gate, up, down]`; one output `down(act(gate(x)) * up(x))`.
    Ffn { activation: GluActivationType },
}

impl GroupKind<'_> {
    pub(crate) fn accepts(&self, members: usize) -> bool {
        match self {
            Self::Dense | Self::Gather { .. } => members >= 2,
            Self::Glu { .. } => members == 2,
            Self::Ffn { .. } => members == 3,
        }
    }
}

/// Format-neutral description of a group member; formats compare keys to find compatible peers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupKey {
    pub format: &'static str,
    pub dtype: DType,
    /// Logical weight shape `[experts, out, in]`, with `experts == 1` for dense layers.
    pub shape: [usize; 3],
    pub calibration: ActivationCalibration,
}

pub type GroupForward =
    fn(&Tensor, &[GroupEntry<'_>], GroupKind<'_>) -> Result<Option<Vec<Tensor>>>;

/// A layer's offer to join a fused group: its key, the format's grouped kernel, and format-private operands.
#[derive(Clone, Debug)]
pub struct GroupMember {
    pub key: GroupKey,
    pub forward: GroupForward,
    pub operand: Arc<dyn std::any::Any + Send + Sync>,
}

#[derive(Debug)]
pub struct GroupEntry<'a> {
    pub method: &'a dyn QuantMethod,
    pub member: Option<GroupMember>,
}

impl GroupEntry<'_> {
    pub fn key(&self) -> Option<&GroupKey> {
        self.member.as_ref().map(|member| &member.key)
    }

    pub fn operand<T: std::any::Any>(&self, format: &str) -> Option<&T> {
        let member = self.member.as_ref()?;
        if member.key.format != format {
            return None;
        }
        member.operand.downcast_ref::<T>()
    }
}

/// Input of a fused GLU projection: separate gate/value tensors or one `[.., 2 * split_size]` tensor.
#[derive(Clone, Copy, Debug)]
pub enum GluInput<'a> {
    Split {
        gate: &'a Tensor,
        value: &'a Tensor,
    },
    Packed {
        input: &'a Tensor,
        split_size: usize,
    },
}

impl GluInput<'_> {
    pub fn halves(&self) -> Result<Option<(Tensor, Tensor)>> {
        match *self {
            Self::Split { gate, value } => Ok(Some((gate.clone(), value.clone()))),
            Self::Packed { input, split_size } => {
                if split_size == 0 || input.dims().last() != split_size.checked_mul(2).as_ref() {
                    return Ok(None);
                }
                Ok(Some((
                    input.narrow(candle_core::D::Minus1, 0, split_size)?,
                    input.narrow(candle_core::D::Minus1, split_size, split_size)?,
                )))
            }
        }
    }
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync + Debug + QuantizedSerde {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized;

    fn dequantize_w(&self) -> Result<Tensor>;

    #[cfg(feature = "metal")]
    fn prepare_metal_kernels(&self, _activation_dtype: DType) -> Result<()> {
        Ok(())
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically casts to the required quantization activation type and back.
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        if let Some(t) = self.quantized_act_type() {
            let original_ty = a.dtype();
            self.forward_raw(&a.to_dtype(t)?)?.to_dtype(original_ty)
        } else {
            self.forward_raw(a)
        }
    }

    /// Raw matmul without dtype casting. Implementors override this.
    /// Callers should use `forward` instead.
    fn forward_raw(&self, a: &Tensor) -> Result<Tensor>;

    /// Routed matmul: inputs `[tokens, 1|topk, in]`, indices `[tokens, topk]`, output `[tokens, topk, out]`.
    /// Weights are `[experts, out, in]`; activation dtype conversion is handled here.
    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if let Some(t) = self.quantized_act_type() {
            let original_ty = a.dtype();
            self.gather_forward_raw(&a.to_dtype(t)?, indices)?
                .to_dtype(original_ty)
        } else {
            self.gather_forward_raw(a, indices)
        }
    }

    /// Raw gather matmul without dtype casting. Implementors override this.
    /// Callers should use `gather_forward` instead.
    fn gather_forward_raw(&self, _a: &Tensor, _indices: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "{} does not support `gather_forward`. Please raise an issue.",
            self.name()
        )
    }

    /// Offer to join a fused group for `xs`; `None` when the format has no grouped kernel or prefers its own forward.
    fn group_member(&self, _xs: &Tensor, _kind: GroupKind<'_>) -> Result<Option<GroupMember>> {
        Ok(None)
    }

    /// Encoded rows `embedding_forward` decodes in place, so the table stays quantized when used as an embedding.
    fn embedding_table(&self) -> Option<EmbeddingTable<'_>> {
        None
    }

    /// Rows `ids` of the weight as `[..ids, hidden]`; formats without a row decoder dequantize the table.
    fn embedding_forward(&self, ids: &Tensor, out: EmbeddingOutput) -> Result<Tensor> {
        embedding::forward(self, ids, out)
    }

    /// Get the underlying QTensor if this is a GGUF quantized layer.
    /// Used for direct kernel access in grouped MoE prefill paths.
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        None
    }

    /// If this is an AFQ layer, return its (w_q, scales, biases, bits, group_size).
    fn afq_inner(&self) -> Option<crate::afq::AfqInner> {
        None
    }

    /// This expert stack in a shared routed GEMV codec, when the format has one.
    fn routed_weights(&self) -> Option<RoutedWeights> {
        None
    }

    fn activation_quantization_scheme(&self) -> Option<ActivationQuantizationScheme> {
        None
    }

    fn activation_quantization_scheme_for(
        &self,
        _a: &Tensor,
    ) -> Option<ActivationQuantizationScheme> {
        self.activation_quantization_scheme()
    }

    fn preferred_activation_scale_layout_for(&self, a: &Tensor) -> Option<ActivationScaleLayout> {
        self.activation_quantization_scheme_for(a)
            .map(|_| ActivationScaleLayout::RowMajor)
    }

    fn activation_calibration(&self) -> ActivationCalibration {
        ActivationCalibration::Dynamic
    }

    fn quantize_activation(&self, _a: &Tensor) -> Result<QuantizedActivation> {
        candle_core::bail!("{} does not support activation quantization", self.name())
    }

    fn forward_quantized(&self, _a: &QuantizedActivation) -> Result<Tensor> {
        candle_core::bail!(
            "{} does not support prequantized activation input",
            self.name()
        )
    }

    /// `self.forward(act(gate) * value)` with the GLU fused into this layer's activation quantization.
    fn forward_glu(
        &self,
        _input: GluInput<'_>,
        _activation: GluActivationType,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// If a quantized method, return the activation dtype.
    fn quantized_act_type(&self) -> Option<DType>;

    /// Weight dtype and device
    fn dtype_and_device(&self) -> (DType, Device);

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams>;

    /// Add a delta weight from LoRA to the weights. This should be prescaled with alpha.
    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>>;

    /// If the quant is backed by a qmatmul.
    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>;

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        None
    }

    /// Dense weights in forward output order, which may differ from the canonical checkpoint order.
    fn unquant_forward_weight(&self) -> Option<Tensor> {
        None
    }

    fn is_dynamic_lora_active(&self) -> bool {
        false
    }

    fn preserve_dynamic_lora(&self, replacement: Arc<dyn QuantMethod>) -> Arc<dyn QuantMethod> {
        replacement
    }

    fn has_bias(&self) -> bool {
        false
    }

    /// Begin tracking stats into an ImatrixLayerStats
    fn begin_track_stats(&self) -> Result<()> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// End tracking stats into an ImatrixLayerStats. Returns the computed imatrix.
    fn end_track_stats(&self) -> Result<Tensor> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// (forward calls, token rows) accumulated by stats tracking, if enabled.
    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        None
    }

    /// Feed routed activations for per-expert stats; called by the owning MoE block, which alone
    /// knows the token-to-expert pairing. No-op unless routed tracking is enabled.
    fn process_routed_stats(&self, _x: &Tensor, _ids: &Tensor) -> Result<()> {
        Ok(())
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        None
    }

    fn dummy_info(&self) -> Option<DummyLayerInfo> {
        None
    }
}

// `output` needs a wrapper adding nothing to the inner output; `weights`/`group` need forward == inner forward.
macro_rules! forward_quant_hooks {
    ($($group:ident),+ : $acc:path) => {
        $($crate::forward_quant_hooks!(@$group $acc);)+
    };
    (@meta $acc:path) => {
        #[cfg(feature = "metal")]
        fn prepare_metal_kernels(&self, activation_dtype: candle_core::DType) -> candle_core::Result<()> {
            match $acc(self) {
                Some(inner) => inner.prepare_metal_kernels(activation_dtype),
                None => Ok(()),
            }
        }


        fn begin_track_stats(&self) -> candle_core::Result<()> {
            match $acc(self) {
                Some(inner) => inner.begin_track_stats(),
                None => candle_core::bail!(
                    "`{}` does not support tracking stats.",
                    $crate::QuantizedSerde::name(self)
                ),
            }
        }

        fn end_track_stats(&self) -> candle_core::Result<candle_core::Tensor> {
            match $acc(self) {
                Some(inner) => inner.end_track_stats(),
                None => candle_core::bail!(
                    "`{}` does not support tracking stats.",
                    $crate::QuantizedSerde::name(self)
                ),
            }
        }

        fn stats_snapshot(&self) -> Option<(usize, usize)> {
            $acc(self)?.stats_snapshot()
        }

        fn process_routed_stats(
            &self,
            x: &candle_core::Tensor,
            ids: &candle_core::Tensor,
        ) -> candle_core::Result<()> {
            match $acc(self) {
                Some(inner) => inner.process_routed_stats(x, ids),
                None => Ok(()),
            }
        }
    };
    (@activation $acc:path) => {
        fn activation_quantization_scheme(&self) -> Option<$crate::ActivationQuantizationScheme> {
            $acc(self)?.activation_quantization_scheme()
        }

        fn activation_quantization_scheme_for(
            &self,
            a: &candle_core::Tensor,
        ) -> Option<$crate::ActivationQuantizationScheme> {
            $acc(self)?.activation_quantization_scheme_for(a)
        }

        fn preferred_activation_scale_layout_for(
            &self,
            a: &candle_core::Tensor,
        ) -> Option<$crate::ActivationScaleLayout> {
            $acc(self)?.preferred_activation_scale_layout_for(a)
        }

        fn activation_calibration(&self) -> $crate::ActivationCalibration {
            $acc(self).map_or($crate::ActivationCalibration::Dynamic, |inner| {
                inner.activation_calibration()
            })
        }

        fn quantize_activation(
            &self,
            a: &candle_core::Tensor,
        ) -> candle_core::Result<$crate::QuantizedActivation> {
            match $acc(self) {
                Some(inner) => inner.quantize_activation(a),
                None => {
                    candle_core::bail!(
                    "{} does not support activation quantization",
                    $crate::QuantizedSerde::name(self)
                )
                }
            }
        }
    };
    (@output $acc:path) => {
        fn forward_quantized(
            &self,
            a: &$crate::QuantizedActivation,
        ) -> candle_core::Result<candle_core::Tensor> {
            match $acc(self) {
                Some(inner) => inner.forward_quantized(a),
                None => candle_core::bail!(
                    "{} does not support prequantized activation input",
                    $crate::QuantizedSerde::name(self)
                ),
            }
        }

        fn forward_glu(
            &self,
            input: $crate::GluInput<'_>,
            activation: $crate::GluActivationType,
        ) -> candle_core::Result<Option<candle_core::Tensor>> {
            match $acc(self) {
                Some(inner) => inner.forward_glu(input, activation),
                None => Ok(None),
            }
        }
    };
    (@weights $acc:path) => {
        fn get_qtensor(&self) -> Option<std::sync::Arc<candle_core::quantized::QTensor>> {
            $acc(self)?.get_qtensor()
        }

        fn afq_inner(&self) -> Option<$crate::AfqInner> {
            $acc(self)?.afq_inner()
        }

        fn routed_weights(&self) -> Option<$crate::RoutedWeights> {
            $acc(self)?.routed_weights()
        }

        fn unquant_forward_weight(&self) -> Option<candle_core::Tensor> {
            $acc(self)?.unquant_forward_weight()
        }
    };
    (@group $acc:path) => {
        fn group_member(
            &self,
            xs: &candle_core::Tensor,
            kind: $crate::GroupKind<'_>,
        ) -> candle_core::Result<Option<$crate::GroupMember>> {
            match $acc(self) {
                Some(inner) => inner.group_member(xs, kind),
                None => Ok(None),
            }
        }
    };
}
pub(crate) use forward_quant_hooks;

/// Grouped kernels skip each member's `forward`, so no member may track stats or run LoRA; bias is format-owned.
pub(crate) fn group_allowed(xs: &Tensor, members: &[&dyn QuantMethod]) -> bool {
    matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32)
        && members
            .iter()
            .all(|m| m.stats_snapshot().is_none() && !m.is_dynamic_lora_active())
}

/// Outputs in member order (one for `Glu`/`Ffn`); `None` means callers run each member themselves.
pub fn forward_group(
    xs: &Tensor,
    members: &[&dyn QuantMethod],
    kind: GroupKind<'_>,
) -> Result<Option<Vec<Tensor>>> {
    if !kind.accepts(members.len()) || !group_allowed(xs, members) {
        return Ok(None);
    }
    if let Some(outputs) = forward_group_shared_activation(xs, members, kind)? {
        return Ok(Some(outputs));
    }
    let entries = members
        .iter()
        .map(|&method| {
            Ok(GroupEntry {
                method,
                member: method.group_member(xs, kind)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, entry) in entries.iter().enumerate() {
        let Some(member) = &entry.member else {
            continue;
        };
        let tried = entries[..index].iter().any(|earlier| {
            earlier
                .key()
                .is_some_and(|key| key.format == member.key.format)
        });
        if tried {
            continue;
        }
        if let Some(outputs) = (member.forward)(xs, &entries, kind)? {
            return Ok(Some(outputs));
        }
    }
    Ok(None)
}

pub(crate) fn forward_group_shared_activation(
    xs: &Tensor,
    members: &[&dyn QuantMethod],
    kind: GroupKind<'_>,
) -> Result<Option<Vec<Tensor>>> {
    match kind {
        GroupKind::Gather { .. } => Ok(None),
        GroupKind::Dense => forward_shared_activation(xs, members),
        GroupKind::Glu { activation } => {
            let Some(outputs) = forward_shared_activation(xs, members)? else {
                return Ok(None);
            };
            let [gate, up] = two_outputs(outputs)?;
            Ok(Some(vec![fused_glu(&gate, &up, activation)?]))
        }
        GroupKind::Ffn { activation } => {
            let Some(outputs) = forward_shared_activation(xs, &members[..2])? else {
                return Ok(None);
            };
            let [gate, up] = two_outputs(outputs)?;
            let down = members[2];
            let input = GluInput::Split {
                gate: &gate,
                value: &up,
            };
            if let Some(output) = forward_fused_glu(down, input, activation)? {
                return Ok(Some(vec![output]));
            }
            Ok(Some(vec![
                down.forward(&fused_glu(&gate, &up, activation)?)?
            ]))
        }
    }
}

pub(crate) fn two_outputs(outputs: Vec<Tensor>) -> Result<[Tensor; 2]> {
    outputs.try_into().map_err(|_| {
        candle_core::Error::msg("shared gate/up projection returned the wrong output count")
    })
}

/// Quantize `a` once and feed it to every member, when all members agree on scheme, calibration and layout.
pub(crate) fn forward_shared_activation(
    a: &Tensor,
    members: &[&dyn QuantMethod],
) -> Result<Option<Vec<Tensor>>> {
    let Some(first) = members.first() else {
        return Ok(Some(Vec::new()));
    };
    if !matches!(a.dtype(), DType::F16 | DType::BF16) {
        return Ok(None);
    }
    let Some(scheme) = first.activation_quantization_scheme_for(a) else {
        return Ok(None);
    };
    if first.preferred_activation_scale_layout_for(a) != Some(ActivationScaleLayout::RowMajor) {
        return Ok(None);
    }
    let calibration = first.activation_calibration();
    if members[1..].iter().any(|member| {
        member.activation_quantization_scheme_for(a) != Some(scheme)
            || member.activation_calibration() != calibration
            || member.preferred_activation_scale_layout_for(a)
                != Some(ActivationScaleLayout::RowMajor)
    }) {
        return Ok(None);
    }
    let activation = first.quantize_activation(a)?;
    if activation.scheme() != scheme {
        candle_core::bail!(
            "{} produced activation quantization scheme {:?}, expected {:?}",
            first.name(),
            activation.scheme(),
            scheme
        );
    }
    if activation.global_scale() != calibration.global_scale() {
        candle_core::bail!(
            "shared activation quantizer returned a different calibrated global scale"
        );
    }
    if activation.scale_layout() != ActivationScaleLayout::RowMajor {
        candle_core::bail!(
            "{} produced activation scale layout {:?}, expected row-major",
            first.name(),
            activation.scale_layout()
        );
    }
    members
        .iter()
        .map(|member| member.forward_quantized(&activation))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// `projection.forward(act(gate) * value)` through the projection's fused GLU quantizer, when it has one.
pub fn forward_fused_glu(
    projection: &dyn QuantMethod,
    input: GluInput<'_>,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    if projection.is_dynamic_lora_active() || projection.stats_snapshot().is_some() {
        return Ok(None);
    }
    if let GluInput::Split { gate, value } = input {
        if !matches!(gate.dtype(), DType::BF16 | DType::F16)
            || gate.dtype() != value.dtype()
            || gate.shape() != value.shape()
            || !gate.device().same_device(value.device())
        {
            return Ok(None);
        }
    }
    projection.forward_glu(input, activation)
}

pub(crate) fn single_output(outputs: Option<Vec<Tensor>>) -> Option<Tensor> {
    outputs.and_then(|mut outputs| outputs.pop())
}

/// Fused gate + up + GLU + down, where a format has a kernel for these weights.
pub fn try_fused_ffn(
    xs: &Tensor,
    gate: &dyn QuantMethod,
    up: &dyn QuantMethod,
    down: &dyn QuantMethod,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    forward_group(xs, &[gate, up, down], GroupKind::Ffn { activation }).map(single_output)
}

/// Fused gate + up + GLU, where a format has a kernel for these weights.
pub fn try_fused_gate_up(
    xs: &Tensor,
    gate: &dyn QuantMethod,
    up: &dyn QuantMethod,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    forward_group(xs, &[gate, up], GroupKind::Glu { activation }).map(single_output)
}

/// Fused Q/K/V projections, where a format has a kernel for these weights.
pub fn try_fused_qkv(
    xs: &Tensor,
    q: &dyn QuantMethod,
    k: &dyn QuantMethod,
    v: &dyn QuantMethod,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let Some(outputs) = forward_group(xs, &[q, k, v], GroupKind::Dense)? else {
        return Ok(None);
    };
    let [q, k, v]: [Tensor; 3] = outputs.try_into().map_err(|_| {
        candle_core::Error::msg("fused QKV projection returned the wrong output count")
    })?;
    Ok(Some((q, k, v)))
}

impl Module for dyn QuantMethod {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        QuantMethod::forward(self, xs)
    }
}
