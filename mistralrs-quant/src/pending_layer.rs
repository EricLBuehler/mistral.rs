use std::{
    borrow::Cow,
    fmt::Debug,
    sync::{
        atomic::AtomicUsize,
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    DistributedKind, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
};

enum PendingState {
    Pending(Receiver<Result<Arc<dyn QuantMethod>>>),
    Ready(Arc<dyn QuantMethod>),
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
    pub fn new(rx: Receiver<Result<Arc<dyn QuantMethod>>>) -> Self {
        Self {
            inner: Mutex::new(PendingState::Pending(rx)),
        }
    }

    /// Block until the background quantization task completes and return the
    /// resolved layer. Subsequent calls return the cached result immediately.
    fn resolve(&self) -> Result<Arc<dyn QuantMethod>> {
        let mut state = self.inner.lock().expect("PendingIsqLayer lock poisoned");
        match &*state {
            PendingState::Ready(layer) => Ok(layer.clone()),
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
                        .map_err(|e| candle_core::Error::Msg(format!("ISQ channel error: {e}")))?;
                    match result {
                        Ok(layer) => {
                            *state = PendingState::Ready(layer.clone());
                            Ok(layer)
                        }
                        Err(e) => {
                            // Leave in Taken state; the error is propagated.
                            Err(e)
                        }
                    }
                } else {
                    unreachable!()
                }
            }
        }
    }
}

impl Debug for PendingIsqLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state_str = match &*self.inner.lock().unwrap() {
            PendingState::Pending(_) => "Pending",
            PendingState::Ready(_) => "Ready",
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

    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        // We must return owned data since we can't borrow from the resolved layer
        let layer = self.resolve()?;
        let data = layer.serialize()?;
        Ok(Cow::Owned(data.into_owned()))
    }

    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let layer = self.resolve()?;
        let data = layer.serialize_with_bias(bias)?;
        Ok(Cow::Owned(data.into_owned()))
    }
}

impl QuantMethod for PendingIsqLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("PendingIsqLayer cannot be created via QuantMethodConfig")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.resolve()?.dequantize_w()
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        self.resolve()?.forward(a)
    }

    fn forward_autocast(&self, a: &Tensor) -> Result<Tensor> {
        self.resolve()?.forward_autocast(a)
    }

    fn gather_forward_autocast(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.resolve()?.gather_forward_autocast(a, indices)
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.resolve()?.gather_forward(a, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        self.resolve().ok()?.quantized_act_type()
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        self.resolve()
            .expect("PendingIsqLayer failed to resolve for dtype_and_device")
            .dtype_and_device()
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

    fn begin_track_stats(&mut self) -> Result<()> {
        // Immediate ISQ is the no-imatrix path, so stats tracking is never used.
        candle_core::bail!("`PendingIsqLayer` does not support tracking stats.")
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        candle_core::bail!("`PendingIsqLayer` does not support tracking stats.")
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        self.resolve().ok()?.is_distributed()
    }
}

pub type IsqSender = mpsc::SyncSender<Result<Arc<dyn QuantMethod>>>;
pub type IsqReceiver = Receiver<Result<Arc<dyn QuantMethod>>>;

/// Create a channel pair for use with `PendingIsqLayer`. The sender should be
/// given to a thread pool task; the receiver is passed to `PendingIsqLayer::new`.
pub fn pending_isq_channel() -> (IsqSender, IsqReceiver) {
    mpsc::sync_channel(1)
}
