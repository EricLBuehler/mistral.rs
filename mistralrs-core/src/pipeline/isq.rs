use std::{
    sync::{atomic::AtomicUsize, Arc},
    time::Instant,
};

use candle_core::{quantized::GgmlDType, Device};
use indicatif::{ProgressBar, ProgressStyle};
use mistralrs_quant::QuantMethod;
use tracing::info;

use crate::device_map::DeviceMapper;

/// Parse ISQ value: one of
/// - `Q4_0`
/// - `Q4_1`
/// - `Q5_0`
/// - `Q5_1`
/// - `Q8_0`
/// - `Q8_1`
/// - `Q2K`
/// - `Q3K`
/// - `Q4K`
/// - `Q5K`
/// - `Q6K`
/// - `Q8K`
pub fn parse_isq_value(s: &str) -> Result<GgmlDType, String> {
    match s.to_lowercase().as_str() {
        "q4_0" => Ok(GgmlDType::Q4_0),
        "q4_1" => Ok(GgmlDType::Q4_1),
        "q5_0" => Ok(GgmlDType::Q5_0),
        "q5_1" => Ok(GgmlDType::Q5_1),
        "q8_0" => Ok(GgmlDType::Q8_0),
        "q8_1" => Ok(GgmlDType::Q8_1),
        "q2k" => Ok(GgmlDType::Q2K),
        "q3k" => Ok(GgmlDType::Q3K),
        "q4k" => Ok(GgmlDType::Q4K),
        "q5k" => Ok(GgmlDType::Q5K),
        "q6k" => Ok(GgmlDType::Q6K),
        "q8k" => Ok(GgmlDType::Q8K),
        _ => Err(format!("GGML type {s} unknown, choose one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`.")),
    }
}

pub trait IsqModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    );
    /// Quantize the model in-situ.
    fn quantize(&mut self, dtype: GgmlDType, device: Device) -> candle_core::Result<()> {
        {
            let (tensors, mapper) = self.get_layers();
            let total_tensors = tensors.len();
            let n_quantized = AtomicUsize::new(0);
            info!("Applying in-situ quantization into {dtype:?} to {total_tensors} tensors.");
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

            let t_start = Instant::now();
            #[cfg(not(feature = "metal"))]
            {
                info!("Applying ISQ on {} threads.", rayon::current_num_threads());

                use indicatif::ParallelProgressIterator;
                use rayon::iter::{
                    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
                };
                tensors
                    .into_par_iter()
                    .zip(devices)
                    .progress_with(bar)
                    .for_each(|((tensor, _), device)| {
                        *tensor = tensor.clone().apply_isq(dtype, &n_quantized).unwrap();
                        device.synchronize().unwrap();
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
                        *tensor = tensor.clone().apply_isq(dtype, &n_quantized).unwrap();
                        device.synchronize().unwrap();
                    });
            }
            let delta = Instant::now().duration_since(t_start).as_secs_f32();
            info!("Applied in-situ quantization into {dtype:?} to {n_quantized:?} tensors out of {total_tensors} total tensors. Took {delta:.2}s", );
        }
        Ok(())
    }
}
