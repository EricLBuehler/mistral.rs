use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fs::File,
    path::PathBuf,
    str::FromStr,
    sync::{atomic::AtomicUsize, Arc},
    time::Instant,
};

use anyhow::Result;
use candle_core::{quantized, Context, Device, Tensor};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use itertools::Itertools;
use mistralrs_quant::{
    FP8Linear, GgufMatMul, HqqLayer, IsqType, QuantMethod, QuantizedSerde, QuantizedSerdeType,
    UnquantLinear,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{device_map::DeviceMapper, topology::LayerTopology, Topology};

pub(crate) const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";

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
        "fp8" => IsqType::F8E4M3,
        // "hqq3" => IsqType::HQQ3,
        // "hqq2" => IsqType::HQQ2,
        // "hqq1" => IsqType::HQQ1,
        _ => return Err(format!("ISQ type {s} unknown, choose one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`, `HQQ8`, `HQQ4`, `FP8`.")),
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
                | IsqType::HQQ4
                | IsqType::F8E4M3 // | IsqType::HQQ3
                                  // | IsqType::HQQ2
                                  // | IsqType::HQQ1
        ) {
            return Err("ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`, `FP8`".to_string());
        }
    }
    Ok(tp)
}

#[derive(Clone, Debug, Copy, Default, Deserialize)]
pub enum IsqOrganization {
    #[default]
    #[serde(rename = "default")]
    Default,
    /// Only quantize MoE experts, if applicable. The enables MoQE.
    /// <https://arxiv.org/abs/2310.02410>
    #[serde(rename = "moqe")]
    MoeExpertsOnly,
}

impl FromStr for IsqOrganization {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "default" => Ok(Self::Default),
            "moqe" => Ok(Self::MoeExpertsOnly),
            other => Err(format!(
                "Expected ISQ organization `default` or `moqe`, got `{other}`"
            )),
        }
    }
}

pub struct UqffFullSer<'a> {
    pub tokenizer: &'a Tokenizer,
    pub template_filename: &'a Option<PathBuf>,
    pub generation_config: Option<&'a PathBuf>,
    pub config: String,
    pub processor_filename: &'a Option<PathBuf>,
    pub preprocessor_filename: &'a Option<PathBuf>,
}

pub enum ImatrixDataSource<'a> {
    File(&'a PathBuf),
    Collected,
}

pub trait IsqModel {
    /// Corresponds to `IsqOrganization::Default`
    #[allow(clippy::type_complexity)]
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    );

    /// This is used for imatrix generation internally. Begin stats tracking.
    fn begin_track_stats(&mut self) -> anyhow::Result<()> {
        let layers = self
            .get_layers()
            .0
            .into_iter()
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();
        for layer in layers {
            Arc::get_mut(layer).unwrap().begin_track_stats()?;
        }
        Ok(())
    }

    /// End stats tracking and return the imatrix data
    fn extract_imatrix_data(&mut self) -> candle_core::Result<HashMap<usize, Option<Vec<f32>>>> {
        let layers = self
            .get_layers()
            .0
            .into_iter()
            .enumerate()
            .map(|(i, (layer, _))| (i, layer))
            .collect::<Vec<_>>();
        let mut data = HashMap::new();
        for (i, layer) in layers {
            data.insert(i, Some(layer.end_track_stats()?.to_vec1::<f32>()?));
        }
        Ok(data)
    }

    /// Corresponds to `IsqOrganization::MoeExpertsOnly`
    /// https://arxiv.org/abs/2310.02410
    #[allow(clippy::type_complexity)]
    fn get_layers_moe_experts_only(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.get_layers()
    }

    /// Corresponds to `IsqOrganization::MoeExpertsOnly`
    /// This is used for imatrix generation internally. Begin stats tracking.
    fn begin_track_stats_moe_experts_only(&mut self) -> anyhow::Result<()> {
        let layers = self
            .get_layers()
            .0
            .into_iter()
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();
        for layer in layers {
            Arc::get_mut(layer).unwrap().begin_track_stats()?;
        }
        Ok(())
    }

    /// Corresponds to `IsqOrganization::MoeExpertsOnly`
    /// End stats tracking and return the imatrix data
    fn extract_imatrix_data_moe_experts_only(
        &mut self,
    ) -> candle_core::Result<HashMap<usize, Option<Vec<f32>>>> {
        let layers = self
            .get_layers()
            .0
            .into_iter()
            .enumerate()
            .map(|(i, (layer, _))| (i, layer))
            .collect::<Vec<_>>();
        let mut data = HashMap::new();
        for (i, layer) in layers {
            data.insert(i, Some(layer.end_track_stats()?.to_vec1::<f32>()?));
        }
        Ok(data)
    }

    /// Corresponding to the specific order the model produces ISQ layers (None means
    /// do not search for in the imatrix file). This is used to pair ISQ layers with the
    /// corresponding imatrix weights.
    ///
    /// - This is only for loading from a llama.cpp imatrix file.
    /// - Corresponds to `IsqOrganization::Default`
    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        // TODO: make this required.
        candle_core::bail!("This model does not support quantizing with an imatrix.");
    }

    /// Residual tensors for generating a UQFF file. Counterpart to [`get_layers`].
    fn residual_tensors(&self) -> Vec<(String, Tensor)>;

    /// Residual tensors for generating a UQFF file. Counterpart to [`get_layers_moe_experts_only`].
    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        None
    }

    /// Quantize the model in-situ.
    ///
    /// This function will also create a UQFF file, or, if the model supports it (residual tensors are returned),
    /// a full serialization is created.
    #[allow(clippy::too_many_arguments)]
    fn quantize(
        &mut self,
        dtype: Option<IsqType>,
        device: Device,
        topology: Option<&Topology>,
        silent: bool,
        imatrix_source: Option<ImatrixDataSource<'_>>,
        organization: IsqOrganization,
        write_artifacts: Option<&PathBuf>,
        full_ser: UqffFullSer<'_>,
    ) -> candle_core::Result<()> {
        {
            let imatrix_to_weight = match imatrix_source {
                Some(ImatrixDataSource::File(imatrix)) => {
                    let mut imatrix_data = quantized::imatrix_file::load_imatrix(imatrix.clone())?;
                    let imatrix_mapping = self
                        .imatrix_names()?
                        .into_iter()
                        .enumerate()
                        .collect::<HashMap<_, _>>();

                    let layer_to_weight = imatrix_mapping
                        .into_iter()
                        .map(|(i, name)| {
                            if let Some(name) = name {
                                (i, Some(imatrix_data.remove(&name).unwrap()))
                            } else {
                                (i, None)
                            }
                        })
                        .collect::<HashMap<_, _>>();
                    info!(
                        "Quantizing with imatrix file `{}`, {} imatrix weights",
                        imatrix.display(),
                        layer_to_weight.iter().filter(|(_, x)| x.is_some()).count()
                    );
                    Some(layer_to_weight)
                }
                Some(ImatrixDataSource::Collected) => {
                    let data = match organization {
                        IsqOrganization::Default => self.extract_imatrix_data()?,
                        IsqOrganization::MoeExpertsOnly => {
                            self.extract_imatrix_data_moe_experts_only()?
                        }
                    };
                    info!(
                        "Quantizing with collected imatrix data, {} imatrix weights",
                        data.iter().filter(|(_, x)| x.is_some()).count()
                    );
                    Some(data)
                }
                None => {
                    // Dummy, just for zip
                    None
                }
            };

            let (mut tensors, mapper) = match organization {
                IsqOrganization::Default => self.get_layers(),
                IsqOrganization::MoeExpertsOnly => self.get_layers_moe_experts_only(),
            };

            let imatrix_to_weight: Vec<Option<Vec<f32>>> =
                if let Some(mut imatrix_to_weight) = imatrix_to_weight {
                    let ordered_keys = imatrix_to_weight
                        .keys()
                        .copied()
                        .sorted()
                        .collect::<Vec<_>>();
                    ordered_keys
                        .into_iter()
                        .map(|layer| imatrix_to_weight.remove(&layer).unwrap())
                        .collect()
                } else {
                    vec![None; tensors.len()]
                };

            let total_tensors = tensors.len();
            let n_quantized = AtomicUsize::new(0);
            if let Some(topology) = topology {
                let mut dtypes = HashSet::new();
                for layer in topology.0.iter().flatten() {
                    if let LayerTopology {
                        isq: Some(isq_dtype),
                        device: _,
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
                    .filter_map(|topo| topo.as_ref().map(|x| (x.isq, x.device.clone())))
                    .collect::<Vec<_>>()
            });

            let mut devices_and_dtypes = Vec::new();
            for (_, layer_num) in &tensors {
                let device = if let Some(ref layers) = layers {
                    if let Some(layer) = layer_num {
                        layers
                            .get(*layer)
                            .as_ref()
                            .map(|x| x.1.clone())
                            .unwrap_or(Some(device.clone()))
                            .unwrap_or(device.clone())
                    } else {
                        device.clone()
                    }
                } else if let Some(layer_num) = layer_num {
                    mapper
                        .device_for(*layer_num, false)
                        .cloned()
                        .unwrap_or(device.clone())
                } else {
                    device.clone()
                };
                let dtype = if let Some(ref layers) = layers {
                    if let Some(layer) = layer_num {
                        layers.get(*layer).cloned().map(|x| x.0).unwrap_or(dtype)
                    } else {
                        dtype
                    }
                } else {
                    dtype
                };
                devices_and_dtypes.push((device, dtype));
            }

            let t_start = Instant::now();

            use rayon::iter::IntoParallelRefIterator;

            // Get the MINIMUM of the max isq threads the quant method allows
            #[cfg(not(feature = "metal"))]
            let mut minimum_max_threads = {
                let current_rayon_threads = rayon::current_num_threads();
                tensors
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
                    .unwrap_or(current_rayon_threads)
            };
            #[cfg(feature = "metal")]
            let mut minimum_max_threads = 1;

            if matches!(imatrix_source, Some(ImatrixDataSource::Collected)) {
                // Collected imatrix means that the model is potentially on the gpu already
                minimum_max_threads = 1;
            }

            info!("Applying ISQ on {minimum_max_threads} threads.");

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(minimum_max_threads)
                .build()
                .map_err(candle_core::Error::msg)?;

            pool.install(|| {
                use indicatif::ParallelProgressIterator;
                use rayon::iter::{
                    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
                };
                if silent {
                    tensors
                        .par_iter_mut()
                        .zip(devices_and_dtypes)
                        .zip(imatrix_to_weight)
                        .for_each(|(((tensor, _), (device, dtype)), imatrix_weight)| {
                            **tensor = tensor
                                .clone()
                                .apply_isq(dtype, device.clone(), &n_quantized, imatrix_weight)
                                .unwrap();
                            device.synchronize().unwrap();
                        });
                } else {
                    tensors
                        .par_iter_mut()
                        .zip(devices_and_dtypes)
                        .zip(imatrix_to_weight)
                        .progress_with(bar)
                        .for_each(|(((tensor, _), (device, dtype)), imatrix_weight)| {
                            **tensor = tensor
                                .clone()
                                .apply_isq(dtype, device.clone(), &n_quantized, imatrix_weight)
                                .unwrap();
                            device.synchronize().unwrap();
                        });
                }
            });

            if let Some(serialized) = write_artifacts {
                info!(
                    "Serializing {total_tensors} ISQ tensors to `{}`.",
                    serialized.display()
                );

                if serialized.extension().is_none_or(|ext| ext != "uqff") {
                    candle_core::bail!("UQFF output path extension must be `.uqff`",);
                }

                let bar = ProgressBar::new(total_tensors as u64);
                bar.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] [{bar:40.red/magenta}] {pos}/{len} ({eta})")
                        .unwrap()
                        .progress_chars("#>-"),
                );

                #[cfg(not(feature = "metal"))]
                let n_threads = 2;
                #[cfg(feature = "metal")]
                let n_threads = 1;

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .map_err(candle_core::Error::msg)?;

                let quantized_values = pool.install(|| {
                    if silent {
                        tensors
                            .par_iter()
                            .enumerate()
                            .filter(|(_, (layer, _))| layer.isq_serde_supported())
                            .map(|(i, (layer, _))| {
                                Ok((
                                    i.to_string(),
                                    Tensor::new(Cow::into_owned(layer.serialize()?), &Device::Cpu)?,
                                ))
                            })
                            .collect::<candle_core::Result<Vec<_>>>()
                    } else {
                        tensors
                            .par_iter()
                            .enumerate()
                            .progress_with(bar)
                            .filter(|(_, (layer, _))| layer.isq_serde_supported())
                            .map(|(i, (layer, _))| {
                                Ok((
                                    i.to_string(),
                                    Tensor::new(Cow::into_owned(layer.serialize()?), &Device::Cpu)?,
                                ))
                            })
                            .collect::<candle_core::Result<Vec<_>>>()
                    }
                });

                let parent = serialized
                    .parent()
                    .context("Target UQFF path must have a filename!")?;

                std::fs::create_dir_all(parent)?;

                safetensors::serialize_to_file(quantized_values?, &None, serialized)?;

                let residual = match organization {
                    IsqOrganization::Default => self.residual_tensors(),
                    IsqOrganization::MoeExpertsOnly => self
                        .residual_tensors_moe_experts_only()
                        .unwrap_or(self.residual_tensors()),
                };

                let residual_out = parent.join(UQFF_RESIDUAL_SAFETENSORS);
                let config_out = parent.join("config.json");
                let tokenizer_out = parent.join("tokenizer.json");
                let tokenizer_cfg_out = parent.join("tokenizer_config.json");
                let gen_cfg_out = parent.join("generation_config.json");
                let processor_out = parent.join("processor_config.json");
                let preprocessor_out = parent.join("preprocessor_config.json");

                info!(
                    "Serializing {} residual tensors to `{}`.",
                    residual.len(),
                    residual_out.display()
                );

                safetensors::serialize_to_file(residual, &None, &residual_out)?;

                let UqffFullSer {
                    tokenizer,
                    template_filename,
                    generation_config,
                    config,
                    processor_filename,
                    preprocessor_filename,
                } = full_ser;

                info!("Serializing configuration to `{}`.", config_out.display());

                std::fs::write(config_out, config)?;

                info!("Serializing tokenizer to `{}`.", tokenizer_out.display());

                serde_json::to_writer_pretty(File::create(&tokenizer_out)?, tokenizer)
                    .map_err(candle_core::Error::msg)?;

                if let Some(template_filename) = template_filename {
                    info!(
                        "Serializing tokenizer config to `{}`.",
                        tokenizer_cfg_out.display()
                    );

                    let template =
                        std::fs::read(template_filename).map_err(candle_core::Error::msg)?;
                    std::fs::write(&tokenizer_cfg_out, template)
                        .map_err(candle_core::Error::msg)?;
                }

                if let Some(generation_config) = generation_config {
                    info!(
                        "Serializing generation config to `{}`.",
                        gen_cfg_out.display()
                    );

                    let cfg = std::fs::read(generation_config).map_err(candle_core::Error::msg)?;
                    std::fs::write(&gen_cfg_out, cfg).map_err(candle_core::Error::msg)?;
                }

                if let Some(processor_config) = processor_filename {
                    info!(
                        "Serializing processor config to `{}`.",
                        processor_out.display()
                    );

                    let cfg = std::fs::read(processor_config).map_err(candle_core::Error::msg)?;
                    std::fs::write(&processor_out, cfg).map_err(candle_core::Error::msg)?;
                }

                if let Some(preprocessor_config) = preprocessor_filename {
                    info!(
                        "Serializing preprocessor config to `{}`.",
                        preprocessor_out.display()
                    );

                    let cfg =
                        std::fs::read(preprocessor_config).map_err(candle_core::Error::msg)?;
                    std::fs::write(&preprocessor_out, cfg).map_err(candle_core::Error::msg)?;
                }
            }
            let delta = Instant::now().duration_since(t_start).as_secs_f32();
            info!("Applied in-situ quantization into {dtype:?} to {n_quantized:?} tensors out of {total_tensors} total tensors. Took {delta:.2}s", );
        }
        Ok(())
    }

    fn load_from_artifacts(
        &mut self,
        device: Device,
        topology: Option<&Topology>,
        silent: bool,
        artifacts: &PathBuf,
    ) -> candle_core::Result<()> {
        let (tensors, mapper) = self.get_layers();
        let total_tensors = tensors.len();

        let layers = topology.map(|x| {
            x.0.iter()
                .filter_map(|topo| topo.as_ref().map(|x| (x.isq, x.device.clone())))
                .collect::<Vec<_>>()
        });

        let mut devices = Vec::new();
        for (_, layer_num) in &tensors {
            let device = if let Some(ref layers) = layers {
                if let Some(layer) = layer_num {
                    layers
                        .get(*layer)
                        .as_ref()
                        .map(|x| x.1.clone())
                        .unwrap_or(Some(device.clone()))
                        .unwrap_or(device.clone())
                } else {
                    device.clone()
                }
            } else if let Some(layer_num) = layer_num {
                mapper
                    .device_for(*layer_num, false)
                    .cloned()
                    .unwrap_or(device.clone())
            } else {
                device.clone()
            };
            devices.push(device);
        }

        let artifacts = unsafe { candle_core::safetensors::MmapedSafetensors::new(artifacts)? };

        let artifact_isqs = artifacts
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                (
                    name.parse::<usize>()
                        .expect("Name should be parseable as usize"),
                    tensor,
                )
            })
            .collect::<HashMap<_, _>>();

        if artifact_isqs.len() != total_tensors {
            candle_core::bail!(
                "Number of artifacts ({}) does not match the number of ISQ layers ({total_tensors})",
                artifact_isqs.len(),
            );
        }

        let bar = ProgressBar::new(total_tensors as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.red/magenta}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let t_start = Instant::now();

        if silent {
            (0..tensors.len())
                .into_par_iter()
                .zip(tensors)
                .map(|(i, (tensor, _))| {
                    if let Some(artifact) = artifact_isqs.get(&i) {
                        let artifact = artifact.data();
                        // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
                        let isq_type = artifact[mistralrs_quant::UQFF_QUANT_TYPE_OFFSET];
                        let deserialized = match QuantizedSerdeType::try_from(isq_type as usize)? {
                            QuantizedSerdeType::Gguf => {
                                GgufMatMul::deserialize(Cow::from(artifact), &devices[i])?
                            }
                            QuantizedSerdeType::Unquant => {
                                UnquantLinear::deserialize(Cow::from(artifact), &devices[i])?
                            }
                            QuantizedSerdeType::Hqq => {
                                HqqLayer::deserialize(Cow::from(artifact), &devices[i])?
                            }
                            QuantizedSerdeType::Fp8 => {
                                FP8Linear::deserialize(Cow::from(artifact), &devices[i])?
                            }
                        };
                        *tensor = deserialized;
                    }
                    Ok(())
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
        } else {
            (0..tensors.len())
                .into_par_iter()
                .zip(tensors)
                .progress_with(bar)
                .map(|(i, (tensor, _))| {
                    if let Some(artifact) = artifact_isqs.get(&i) {
                        let artifact = artifact.data();
                        // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
                        let isq_type = artifact[4];
                        let deserialized = match QuantizedSerdeType::try_from(isq_type as usize)? {
                            QuantizedSerdeType::Gguf => {
                                GgufMatMul::deserialize(Cow::from(artifact), &devices[i])?
                            }
                            QuantizedSerdeType::Unquant => {
                                UnquantLinear::deserialize(Cow::from(artifact), &devices[i])?
                            }
                            QuantizedSerdeType::Hqq => {
                                HqqLayer::deserialize(Cow::from(artifact), &devices[i])?
                            }
                            QuantizedSerdeType::Fp8 => {
                                FP8Linear::deserialize(Cow::from(artifact), &devices[i])?
                            }
                        };
                        *tensor = deserialized;
                    }
                    Ok(())
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
        }

        let delta = Instant::now().duration_since(t_start).as_secs_f32();
        info!("Loaded in-situ quantization artifacts into {total_tensors} total tensors. Took {delta:.2}s", );

        Ok(())
    }
}

/// Trait for loading models with ISQ.
pub(crate) trait IsqModelLoader {
    /// Regex to match layers which will have standard ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}
