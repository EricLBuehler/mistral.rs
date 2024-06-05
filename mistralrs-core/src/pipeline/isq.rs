use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    Device, Tensor,
};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, warn};

use crate::device_map::DeviceMapper;

pub enum QuantizationBehaviour {
    Quantize(GgmlDType),
    Skip,
}

/// Return the fallback dtype for the given dtype.
fn get_fallback(dtype: GgmlDType) -> QuantizationBehaviour {
    // The normal `Q` quants are a bit more lenient than the `K` quants.
    // => Try to fallback to a similar `Q` quant.
    // If that's not possible, skip this tensor.
    match dtype {
        GgmlDType::Q2K => QuantizationBehaviour::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q3K => QuantizationBehaviour::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q4K => QuantizationBehaviour::Quantize(GgmlDType::Q4_1),
        GgmlDType::Q5K => QuantizationBehaviour::Quantize(GgmlDType::Q5_0),
        GgmlDType::Q6K => QuantizationBehaviour::Quantize(GgmlDType::Q5_1),
        GgmlDType::Q8K => QuantizationBehaviour::Quantize(GgmlDType::Q8_1),
        _ => QuantizationBehaviour::Skip,
    }
}

/// Check if the tensor can be quantized with the given dtype.
fn can_quantize(tensor: &Tensor, dtype: GgmlDType) -> bool {
    let dims = tensor.shape().dims();
    // The tensor must not be empty and the last dimension must be a multiple of the block size.
    !(dims.is_empty() || (dims[dims.len() - 1] % dtype.block_size() != 0))
}

/// Check if we should quantize the tensor and if so, with which dtype.
fn get_quantization_behaviour(tensor: &Tensor, dtype: GgmlDType) -> QuantizationBehaviour {
    if dtype == GgmlDType::F32 {
        return QuantizationBehaviour::Skip;
    }

    if can_quantize(tensor, dtype) {
        return QuantizationBehaviour::Quantize(dtype);
    }
    let fallback = get_fallback(dtype);
    match fallback {
        QuantizationBehaviour::Skip => fallback,
        QuantizationBehaviour::Quantize(new_dtype) => get_quantization_behaviour(tensor, new_dtype),
    }
}

macro_rules! generate_isq {
    ($tensor:expr, $device:expr, $dtype:expr, $n_quantized:expr) => {
        if let QMatMul::Tensor(t) = $tensor {
            let t = t.to_device(&$device).unwrap();
            let quantization_behaviour = get_quantization_behaviour(&t, $dtype);
            *$tensor =  match quantization_behaviour{
                QuantizationBehaviour::Skip => {
                    let shape = t.shape();
                    warn!("Skipping quantization of tensor with shape {shape:?} as it is not quantizable.");
                    QMatMul::QTensor(Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap()))
                },
                QuantizationBehaviour::Quantize(dtype) => {
                    $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    QMatMul::QTensor(Arc::new(QTensor::quantize(&t, dtype).unwrap()))
                }
            }
        }
    };
}

pub trait IsqModel {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper);
    /// Quantize the model in-situ.
    fn quantize(&mut self, dtype: GgmlDType, device: Device) -> candle_core::Result<()> {
        let (tensors, mapper) = self.get_tensors();
        let total_tensors = tensors.len();
        let n_quantized = AtomicUsize::new(0);
        info!(
            "Applying in-situ quantization into {dtype:?} to {total_tensors} tensors in parallel."
        );
        let bar = ProgressBar::new(total_tensors as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut devices = Vec::new();
        for (_, layer) in &tensors {
            let device = if let Some(layer) = layer {
                mapper.device_for(*layer, false).unwrap_or(&device)
            } else {
                &device
            };
            devices.push(device.clone());
        }

        #[cfg(not(feature = "metal"))]
        {
            use indicatif::ParallelProgressIterator;
            use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
            tensors
                .into_par_iter()
                .zip(devices)
                .progress_with(bar)
                .for_each(|((tensor, _), device)| {
                    generate_isq!(tensor, device, dtype, n_quantized)
                });
        }

        #[cfg(feature = "metal")]
        {
            use indicatif::ProgressIterator;
            tensors
                .into_iter()
                .zip(devices)
                .progress_with(bar)
                .for_each(|((tensor, _), device)| {
                    generate_isq!(tensor, device, dtype, n_quantized)
                });
        }
        info!("Applied in-situ quantization into {dtype:?} to {n_quantized:?} tensors out of {total_tensors} total tensors.");

        Ok(())
    }
}
