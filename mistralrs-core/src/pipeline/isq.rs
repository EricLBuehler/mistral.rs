use std::{
    collections::HashSet,
    sync::{atomic::AtomicUsize, Arc},
    time::Instant,
};

use candle_core::Device;
use indicatif::{ProgressBar, ProgressStyle};
use mistralrs_quant::{IsqType, QuantMethod};
use tracing::info;

use crate::{device_map::DeviceMapper, topology::LayerTopology, Topology};

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
/// - `HQQ1`
/// - `HQQ2`
/// - `HQQ3`
/// - `HQQ4`
/// - `HQQ8`
pub fn parse_isq_value(s: &str) -> Result<IsqType, String> {
    let tp = match s.to_lowercase().as_str() {
        "q4_0" => IsqType::Q4_0,
        "q4_1" => IsqType::Q4_1,
        "q5_0" => IsqType::Q5_0,
        "q5_1" => IsqType::Q5_1,
        "q8_0" => IsqType::Q8_0,
        "q8_1" => IsqType::Q8_1,
        "q2k" => IsqType::Q2K,
        "q3k" => IsqType::Q3K,
        "q4k" => IsqType::Q4K,
        "q5k" => IsqType::Q5K,
        "q6k" => IsqType::Q6K,
        "q8k" => IsqType::Q8K,
        "hqq8" => IsqType::HQQ8,
        "hqq4" => IsqType::HQQ4,
        // "hqq3" => IsqType::HQQ3,
        // "hqq2" => IsqType::HQQ2,
        // "hqq1" => IsqType::HQQ1,
        _ => return Err(format!("ISQ type {s} unknown, choose one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`, `HQQ8`, `HQQ4`.")),
    };
    #[cfg(feature = "cuda")]
    {
        if !matches!(
            tp,
            IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q8_0
                | IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q5K
                | IsqType::Q6K
                | IsqType::HQQ8
                | IsqType::HQQ4 // | IsqType::HQQ3
                                // | IsqType::HQQ2
                                // | IsqType::HQQ1
        ) {
            return Err("GGML ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`".to_string());
        }
    }
    Ok(tp)
}

pub trait IsqModel {
    #[allow(clippy::type_complexity)]
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    );
    /// Quantize the model in-situ.
    fn quantize(
        &mut self,
        dtype: Option<IsqType>,
        device: Device,
        topology: Option<&Topology>,
    ) -> candle_core::Result<()> {
        {
            let (tensors, mapper) = self.get_layers();
            let total_tensors = tensors.len();
            let n_quantized = AtomicUsize::new(0);
            if let Some(topology) = topology {
                let mut dtypes = HashSet::new();
                for layer in topology.0.iter().flatten() {
                    if let LayerTopology {
                        isq: Some(isq_dtype),
                    } = layer
                    {
                        dtypes.insert(isq_dtype);
                    }
                }
                info!("Applying in-situ quantization into {:?} to {total_tensors} tensors according to topology.", dtypes.into_iter().collect::<Vec<_>>());
            } else {
                info!("Applying in-situ quantization into {dtype:?} to {total_tensors} tensors.");
            }
            let bar = ProgressBar::new(total_tensors as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );

            let layers = topology.map(|x| {
                x.0.iter()
                    .filter_map(|topo| topo.as_ref().map(|x| x.isq))
                    .collect::<Vec<_>>()
            });

            let mut devices_and_dtypes = Vec::new();
            for (_, layer) in &tensors {
                let device = if let Some(layer) = layer {
                    mapper.device_for(*layer, false).unwrap_or(&device)
                } else {
                    &device
                };
                let dtype = if let Some(ref layers) = layers {
                    if let Some(layer) = layer {
                        layers.get(*layer).cloned().unwrap_or(dtype)
                    } else {
                        dtype
                    }
                } else {
                    dtype
                };
                devices_and_dtypes.push((device.clone(), dtype));
            }

            let t_start = Instant::now();
            #[cfg(not(feature = "metal"))]
            {
                let current_rayon_threads = rayon::current_num_threads();
                // Get the MINIMUM of the max isq threads the quant method allows
                let minimum_max_threads = tensors
                    .iter()
                    .map(|(q, _)| {
                        if let Some(dtype) = dtype {
                            q.get_max_isq_cpu_threads(dtype)
                                .map(usize::from)
                                .unwrap_or(current_rayon_threads)
                        } else {
                            current_rayon_threads
                        }
                    })
                    .min()
                    .unwrap_or(current_rayon_threads);

                info!("Applying ISQ on {minimum_max_threads} threads.");

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(minimum_max_threads)
                    .build()
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                pool.install(|| {
                    use indicatif::ParallelProgressIterator;
                    use rayon::iter::{
                        IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
                    };
                    tensors
                        .into_par_iter()
                        .zip(devices_and_dtypes)
                        .progress_with(bar)
                        .for_each(|((tensor, _), (device, dtype))| {
                            *tensor = tensor
                                .clone()
                                .apply_isq(dtype, device.clone(), &n_quantized)
                                .unwrap();
                            device.synchronize().unwrap();
                        });
                });
            }

            #[cfg(feature = "metal")]
            {
                use indicatif::ProgressIterator;
                tensors
                    .into_iter()
                    .zip(devices)
                    .progress_with(bar)
                    .for_each(|((tensor, _), (device, dtype))| {
                        *tensor = tensor
                            .clone()
                            .apply_isq(dtype, device.clone(), &n_quantized)
                            .unwrap();
                        device.synchronize().unwrap();
                    });
            }
            let delta = Instant::now().duration_since(t_start).as_secs_f32();
            info!("Applied in-situ quantization into {dtype:?} to {n_quantized:?} tensors out of {total_tensors} total tensors. Took {delta:.2}s", );
        }
        Ok(())
    }
}
