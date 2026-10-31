use std::{
    fmt::Debug,
    sync::{
        atomic::AtomicUsize,
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    DistributedKind, IsqJobOutput, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde,
};

enum PendingState {
    Pending(Receiver<Result<IsqJobOutput<Arc<dyn QuantMethod>>>>),
    Ready(Arc<dyn QuantMethod>),
    Failed(Arc<candle_core::Error>),
    /// Transitional state used during resolve to avoid holding both the old
    /// receiver and the new layer simultaneously.
    Taken,
}

/// A wrapper around a `QuantMethod` that resolves lazily from a background
/// quantization task. Created by `apply_immediate_isq` when a thread pool is
/// available for parallel immediate ISQ.
pub struct PendingIsqLayer {
    inner: Mutex<PendingState>,
}

impl PendingIsqLayer {
    pub fn new(rx: Receiver<Result<IsqJobOutput<Arc<dyn QuantMethod>>>>) -> Self {
        Self {
            inner: Mutex::new(PendingState::Pending(rx)),
        }
    }

    /// Replace the inner layer; propagates to every holder, including the live model.
    pub fn replace(&self, layer: Arc<dyn QuantMethod>) {
        *self.inner.lock().expect("PendingIsqLayer lock poisoned") = PendingState::Ready(layer);
    }

    /// Block until the background quantization task completes and return the
    /// resolved layer. Subsequent calls return the cached result immediately.
    pub fn resolve(&self) -> Result<Arc<dyn QuantMethod>> {
        let mut state = self.inner.lock().expect("PendingIsqLayer lock poisoned");
        match &*state {
            PendingState::Ready(layer) => Ok(layer.clone()),
            PendingState::Failed(error) => Err(candle_core::Error::wrap(error.clone())),
            PendingState::Taken => {
                candle_core::bail!("PendingIsqLayer is in an invalid transitional state")
            }
            PendingState::Pending(_) => {
                // Take the receiver out so we can receive without holding the
                // lock on the enum variant (swap to Taken first).
                let old = std::mem::replace(&mut *state, PendingState::Taken);
                if let PendingState::Pending(rx) = old {
                    let result = rx
                        .recv()
                        .map_err(|e| candle_core::Error::Msg(format!("ISQ channel error: {e}")))
                        .and_then(|result| result);
                    match result {
                        Ok(output) => {
                            let layer = output.value;
                            *state = PendingState::Ready(layer.clone());
                            Ok(layer)
                        }
                        Err(e) => {
                            let error = Arc::new(e);
                            *state = PendingState::Failed(error.clone());
                            Err(candle_core::Error::wrap(error))
                        }
                    }
                } else {
                    unreachable!()
                }
            }
        }
    }
}

impl PendingIsqLayer {
    fn resolved(&self) -> Option<Arc<dyn QuantMethod>> {
        self.resolve().ok()
    }
}

impl Debug for PendingIsqLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state_str = match &*self.inner.lock().unwrap() {
            PendingState::Pending(_) => "Pending",
            PendingState::Ready(_) => "Ready",
            PendingState::Failed(_) => "Failed",
            PendingState::Taken => "Taken",
        };
        write!(f, "PendingIsqLayer({state_str})")
    }
}

impl QuantizedSerde for PendingIsqLayer {
    fn name(&self) -> &'static str {
        "pending-isq"
    }

    fn isq_serde_supported(&self) -> bool {
        match self.resolve() {
            Ok(layer) => layer.isq_serde_supported(),
            Err(_) => false,
        }
    }

    fn uqff_type(&self) -> Option<IsqType> {
        self.resolve().ok()?.uqff_type()
    }

    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<crate::UqffTensor>> {
        self.resolve()?.serialize_uqff(prefix, ty)
    }
}

impl QuantMethod for PendingIsqLayer {
    crate::forward_quant_hooks!(weights: Self::resolved);

    #[cfg(feature = "metal")]
    fn prepare_metal_kernels(&self, dtype: DType) -> Result<()> {
        self.resolve()?.prepare_metal_kernels(dtype)
    }

    fn begin_track_stats(&self) -> Result<()> {
        self.resolve()?.begin_track_stats()
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        self.resolve()?.end_track_stats()
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.resolved()?.stats_snapshot()
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.resolve()?.process_routed_stats(x, ids)
    }

    fn activation_quantization_scheme(&self) -> Option<crate::ActivationQuantizationScheme> {
        self.resolved()?.activation_quantization_scheme()
    }

    fn activation_quantization_scheme_for(
        &self,
        x: &Tensor,
    ) -> Option<crate::ActivationQuantizationScheme> {
        self.resolved()?.activation_quantization_scheme_for(x)
    }

    fn preferred_activation_scale_layout_for(
        &self,
        x: &Tensor,
    ) -> Option<crate::ActivationScaleLayout> {
        self.resolved()?.preferred_activation_scale_layout_for(x)
    }

    fn activation_calibration(&self) -> crate::ActivationCalibration {
        self.resolved()
            .map_or(crate::ActivationCalibration::Dynamic, |inner| {
                inner.activation_calibration()
            })
    }

    fn quantize_activation(&self, x: &Tensor) -> Result<crate::QuantizedActivation> {
        self.resolve()?.quantize_activation(x)
    }

    fn forward_quantized(&self, x: &crate::QuantizedActivation) -> Result<Tensor> {
        self.resolve()?.forward_quantized(x)
    }

    fn forward_glu(
        &self,
        input: crate::GluInput<'_>,
        activation: crate::GluActivationType,
    ) -> Result<Option<Tensor>> {
        self.resolve()?.forward_glu(input, activation)
    }

    fn group_member(
        &self,
        x: &Tensor,
        kind: crate::GroupKind<'_>,
    ) -> Result<Option<crate::GroupMember>> {
        self.resolve()?.group_member(x, kind)
    }

    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("PendingIsqLayer cannot be created via QuantMethodConfig")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.resolve()?.dequantize_w()
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        self.resolve()?.forward_raw(a)
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        self.resolve()?.forward(a)
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.resolve()?.gather_forward(a, indices)
    }

    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.resolve()?.gather_forward_raw(a, indices)
    }

    fn embedding_forward(&self, ids: &Tensor, out: crate::EmbeddingOutput) -> Result<Tensor> {
        self.resolve()?.embedding_forward(ids, out)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        self.resolve().ok()?.quantized_act_type()
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        self.resolve()
            .expect("PendingIsqLayer failed to resolve for dtype_and_device")
            .dtype_and_device()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        self.resolve()?.plan_isq(request)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        self.resolve()?.add_delta_w(delta)
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        self.resolve()?
            .clone()
            .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        self.resolve().ok()?.unquant_weight_bias()
    }

    fn is_dynamic_lora_active(&self) -> bool {
        self.resolve()
            .is_ok_and(|layer| layer.is_dynamic_lora_active())
    }

    fn has_bias(&self) -> bool {
        match self.resolve() {
            Ok(layer) => layer.has_bias(),
            Err(_) => false,
        }
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        self.resolve().ok()?.is_distributed()
    }

    fn dummy_info(&self) -> Option<crate::DummyLayerInfo> {
        self.resolve().ok()?.dummy_info()
    }
}

pub type IsqSender = mpsc::SyncSender<Result<IsqJobOutput<Arc<dyn QuantMethod>>>>;
pub type IsqReceiver = Receiver<Result<IsqJobOutput<Arc<dyn QuantMethod>>>>;

/// Create a channel pair for use with `PendingIsqLayer`. The sender should be
/// given to a thread pool task; the receiver is passed to `PendingIsqLayer::new`.
pub fn pending_isq_channel() -> (IsqSender, IsqReceiver) {
    mpsc::sync_channel(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_resolution_survives_capability_queries_and_fallible_hooks() -> Result<()> {
        let (sender, receiver) = pending_isq_channel();
        sender
            .send(Err(candle_core::Error::msg("quantization failed")))
            .unwrap();
        let layer = PendingIsqLayer::new(receiver);
        let input = Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?;
        let ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?;
        assert!(layer.stats_snapshot().is_none());
        for error in [
            layer.process_routed_stats(&input, &ids).unwrap_err(),
            layer.begin_track_stats().unwrap_err(),
            layer.forward(&input).unwrap_err(),
        ] {
            assert!(error.to_string().contains("quantization failed"), "{error}");
        }
        Ok(())
    }
}
