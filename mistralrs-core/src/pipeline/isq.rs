use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    env,
    fs::File,
    path::PathBuf,
    str::FromStr,
    sync::{atomic::AtomicUsize, Arc},
    time::Instant,
};
/// Wrapper around a `Cow<'a, [u8]>` buffer that implements
/// `safetensors::tensor::View`.
///
/// *Purpose*: lets us pass raw byte buffers to
/// `safetensors::serialize_to_file` without cloning them into a `Vec<u8>` or
/// converting to a higher‑level tensor type.  
/// We expose the buffer as a 1‑D `u8` tensor of shape `[len]`.
#[derive(Clone)]
pub struct CowBytesView<'a> {
    data: Cow<'a, [u8]>,
    shape: [usize; 1],
}

impl<'a> CowBytesView<'a> {
    /// Convenience constructor.
    pub fn new(data: Cow<'a, [u8]>) -> Self {
        let len = data.len();
        Self { data, shape: [len] }
    }
}

impl safetensors::tensor::View for CowBytesView<'_> {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        // Serialize as raw bytes
        safetensors::tensor::Dtype::U8
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        assert!(matches!(self.data, Cow::Borrowed(_)));
        // Cloning a `Cow` is cheap (only clones the enum, not the data).
        self.data.clone()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

use anyhow::Result;
use candle_core::{quantized, Context, Device, Tensor};
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use itertools::Itertools;
use mistralrs_quant::{
    AfqLayer, CollectedImatrixData, ColumnParallelLayer, DistributedKind, F8Q8Linear, FP8Linear,
    GgufMatMul, HqqLayer, IsqType, QuantMethod, QuantizeOntoGuard, QuantizedSerde,
    QuantizedSerdeType, ReplicatedLayer, RowParallelLayer, UnquantLinear,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::{
    device_map::DeviceMapper, pipeline::EmbeddingModulePaths, topology::LayerTopology,
    utils::progress::configure_progress_bar, Topology,
};

pub(crate) const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";
// 10 GB max per file
const MAX_UQFF_SIZE_BYTES: usize = 10 * 1024 * 1024 * 1024;
pub const UQFF_MULTI_FILE_DELIMITER: &str = ";";

/// Parse ISQ value.
///
/// If the provided value is a valid integer (one of 2,3,4,5,6,8), the best quantization type will be chosen.
/// Note that the fallback is always a Q/K quantization but on Metal 2,3,4,6,8 uses the fast AFQ.
///
/// One of:
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
/// - `AFQ2`
/// - `AFQ3`
/// - `AFQ4`
/// - `AFQ6`
/// - `AFQ8`
pub fn parse_isq_value(s: &str, device: Option<&Device>) -> Result<IsqType, String> {
    let is_metal = device.map(|device| device.is_metal()).unwrap_or(false);
    let tp = match s.to_lowercase().as_str() {
        "2" if is_metal => IsqType::AFQ2,
        "2" if !is_metal => IsqType::Q2K,
        "3" if is_metal => IsqType::AFQ3,
        "3" if !is_metal => IsqType::Q3K,
        "4" if is_metal => IsqType::AFQ4,
        "4" if !is_metal => IsqType::Q4K,
        "5" => IsqType::Q5K,
        "6" if is_metal => IsqType::AFQ6,
        "6" if !is_metal => IsqType::Q6K,
        "8" if is_metal => IsqType::AFQ8,
        "8" if !is_metal => IsqType::Q8_0,
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
        "afq8" => IsqType::AFQ8,
        "afq6" => IsqType::AFQ6,
        "afq4" => IsqType::AFQ4,
        "afq3" => IsqType::AFQ3,
        "afq2" => IsqType::AFQ2,
        "f8q8" => IsqType::F8Q8,
        // "hqq3" => IsqType::HQQ3,
        // "hqq2" => IsqType::HQQ2,
        // "hqq1" => IsqType::HQQ1,
        _ => return Err(format!("ISQ type {s} unknown, choose one of `2`, `3`, `4`, `6`, `8`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`, `HQQ8`, `HQQ4`, `FP8`, `AFQ8`, `AFQ6`, `AFQ4`, `AFQ3`, `AFQ2`, `F8Q8`.")),
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
                | IsqType::F8E4M3
                | IsqType::AFQ2
                | IsqType::AFQ3
                | IsqType::AFQ4
                | IsqType::AFQ6
                | IsqType::AFQ8
                | IsqType::F8Q8 // | IsqType::HQQ3
                                // | IsqType::HQQ2
                                // | IsqType::HQQ1
        ) {
            return Err("ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`, `FP8`, `AFQ8`, `AFQ6`, `AFQ4`, `AFQ3`, `AFQ2`, `F8Q8`".to_string());
        }
    }
    Ok(tp)
}

/// Given a UQFF filename like `"q4k-0.uqff"`, returns `Some(("q4k", 0))`.
/// Returns `None` for non-sharded filenames like `"model.uqff"` where the
/// suffix after the last `-` is not a number.
pub fn parse_uqff_shard(filename: &str) -> Option<(String, u64)> {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())?;
    let (prefix, suffix) = stem.rsplit_once('-')?;
    let index = suffix.parse::<u64>().ok()?;
    Some((prefix.to_string(), index))
}

/// Expand a single UQFF filename to include all sibling shards.
///
/// Given `"q4k-0.uqff"` and a list of available files, returns
/// `["q4k-0.uqff", "q4k-1.uqff", ...]` for all sequential indices found.
/// Non-sharded filenames (those not matching `{prefix}-{N}.uqff`) are returned as-is.
pub fn expand_uqff_shards(first_file: &str, available_files: &[String]) -> Vec<String> {
    let Some((prefix, _)) = parse_uqff_shard(first_file) else {
        return vec![first_file.to_string()];
    };
    let mut shards = Vec::new();
    for index in 0u64.. {
        let candidate = format!("{prefix}-{index}.uqff");
        if available_files.iter().any(|f| f == &candidate) {
            shards.push(candidate);
        } else {
            break;
        }
    }
    if shards.is_empty() {
        vec![first_file.to_string()]
    } else {
        shards
    }
}

#[derive(Clone, Debug, Copy, Default, Deserialize, serde::Serialize)]
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
    pub modules: Option<&'a String>,
    pub module_paths: Option<&'a [EmbeddingModulePaths]>,
    pub generation_config: Option<&'a PathBuf>,
    pub config: String,
    pub processor_filename: &'a Option<PathBuf>,
    pub preprocessor_filename: &'a Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
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
    fn extract_imatrix_data(&mut self) -> candle_core::Result<CollectedImatrixData> {
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
        Ok(CollectedImatrixData(data))
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
    ) -> candle_core::Result<CollectedImatrixData> {
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
        Ok(CollectedImatrixData(data))
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
        apply_quantization: bool,
        write_artifacts: Option<&PathBuf>,
        full_ser: UqffFullSer<'_>,
        multi_progress: Arc<MultiProgress>,
    ) -> candle_core::Result<()> {
        {
            let mut imatrix_source = imatrix_source;
            let mut imatrix_to_weight_map: Option<HashMap<usize, Option<Vec<f32>>>> =
                if apply_quantization {
                    match imatrix_source.take() {
                        Some(ImatrixDataSource::File(imatrix)) => {
                            let ext = imatrix.extension().ok_or(candle_core::Error::msg(
                                "Expected an extension for the imatrix source file.",
                            ))?;
                            if ext == "cimatrix" {
                                info!(
                                    "Loading collected imatrix source file: `{}`",
                                    imatrix.display()
                                );
                                let data = CollectedImatrixData::load_imatrix(imatrix)?;
                                info!(
                                    "Quantizing with collected imatrix data, {} imatrix weights",
                                    data.0.iter().filter(|(_, x)| x.is_some()).count()
                                );
                                Some(data.0)
                            } else {
                                if ext != "imatrix" {
                                    warn!("Imatrix source file extension is {ext:?}, expected .imatrix/.cimatrix. Assuming GGUF specification");
                                }
                                info!(
                                    "Loading GGUF-format imatrix source file: `{}`",
                                    imatrix.display()
                                );
                                let mut imatrix_data =
                                    quantized::imatrix_file::load_imatrix(imatrix.clone())?;
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
                        }
                        Some(ImatrixDataSource::Collected) => {
                            let data = match organization {
                                IsqOrganization::Default => self.extract_imatrix_data()?,
                                IsqOrganization::MoeExpertsOnly => {
                                    self.extract_imatrix_data_moe_experts_only()?
                                }
                            };
                            // Save the collected imatrix data so users can reuse it
                            let count = data.0.iter().filter(|(_, x)| x.is_some()).count();
                            let save_path = format!("collected-{count}.cimatrix");
                            info!("Saving collected imatrix data to `{save_path}`");
                            data.save_imatrix(save_path)?;
                            info!(
                                "Quantizing with collected imatrix data, {count} imatrix weights"
                            );
                            Some(data.0)
                        }
                        None => None,
                    }
                } else {
                    if imatrix_source.is_some() {
                        info!("Imatrix source provided but quantization disabled; ignoring input.");
                    }
                    None
                };

            let (mut tensors, mapper) = match organization {
                IsqOrganization::Default => self.get_layers(),
                IsqOrganization::MoeExpertsOnly => self.get_layers_moe_experts_only(),
            };

            let total_tensors = tensors.len();

            if apply_quantization {
                let imatrix_to_weight: Vec<Option<Vec<f32>>> =
                    if let Some(mut imatrix_to_weight) = imatrix_to_weight_map.take() {
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

                let n_quantized = AtomicUsize::new(0);
                if let Some(topology) = topology {
                    let mut dtypes = HashSet::new();
                    for layer in topology.layers.iter().flatten() {
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
                    info!(
                        "Applying in-situ quantization into {dtype:?} to {total_tensors} tensors."
                    );
                }
                let bar = ProgressBar::new(total_tensors as u64);
                configure_progress_bar(&bar);
                bar.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                        .unwrap()
                        .progress_chars("#>-"),
                );
                multi_progress.add(bar.clone());

                let layers = topology.map(|x| {
                    x.layers
                        .iter()
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

                // Get the MINIMUM of the max isq threads the quant method
                let mut minimum_max_threads = {
                    let current_rayon_threads = rayon::current_num_threads();
                    if let Some(dtype) = dtype {
                        dtype
                            .get_max_isq_cpu_threads()
                            .map(usize::from)
                            .unwrap_or(current_rayon_threads)
                    } else {
                        current_rayon_threads
                    }
                };
                if env::var("MISTRALRS_ISQ_SINGLETHREAD").is_ok() {
                    minimum_max_threads = 1;
                }

                if matches!(imatrix_source, Some(ImatrixDataSource::Collected)) {
                    // Collected imatrix means that the model is potentially on the gpu already
                    minimum_max_threads = 1;
                }

                info!("Applying ISQ on {minimum_max_threads} threads.");

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(minimum_max_threads)
                    .build()
                    .map_err(candle_core::Error::msg)?;

                let guard = QuantizeOntoGuard::new();

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
                                    .apply_isq(
                                        dtype,
                                        device.clone(),
                                        &n_quantized,
                                        imatrix_weight,
                                        guard.clone(),
                                    )
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
                                    .apply_isq(
                                        dtype,
                                        device.clone(),
                                        &n_quantized,
                                        imatrix_weight,
                                        guard.clone(),
                                    )
                                    .unwrap();
                                device.synchronize().unwrap();
                            });
                    }
                });

                let t_end = Instant::now();
                info!(
                    "Finished quantization pass in {:.2}s ({} tensors).",
                    t_end.duration_since(t_start).as_secs_f32(),
                    total_tensors
                );
            } else if imatrix_source.is_some() {
                info!(
                    "Imatrix data provided but quantization was skipped; existing tensors will be serialized as-is."
                );
            } else if write_artifacts.is_some() {
                info!(
                    "Skipping additional quantization; serializing {total_tensors} existing tensors."
                );
            }

            if let Some(serialized) = write_artifacts {
                info!(
                    "Serializing {total_tensors} ISQ tensors to `{}`.",
                    serialized.display()
                );

                if serialized.extension().is_none_or(|ext| ext != "uqff") {
                    candle_core::bail!("UQFF output path extension must be `.uqff`",);
                }

                let bar = ProgressBar::new(total_tensors as u64);
                configure_progress_bar(&bar);
                bar.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] [{bar:40.red/magenta}] {pos}/{len} ({eta})")
                        .unwrap()
                        .progress_chars("#>-"),
                );

                // Metal and CUDA require serialization on the current thread because GPU contexts are thread-local.
                // Using a rayon thread pool (even with n_threads=1) creates a new thread without the GPU context.
                #[cfg(any(feature = "metal", feature = "cuda"))]
                let quantized_values: candle_core::Result<Vec<_>> = {
                    tensors
                        .iter()
                        .enumerate()
                        .filter(|(_, (layer, _))| layer.isq_serde_supported())
                        .map(|(i, (layer, _))| {
                            if !silent {
                                bar.inc(1);
                            }
                            Ok((
                                i.to_string(),
                                match layer.serialize()? {
                                    Cow::Borrowed(_) => unreachable!(),
                                    Cow::Owned(owned) => owned,
                                },
                            ))
                        })
                        .collect()
                };

                #[cfg(not(any(feature = "metal", feature = "cuda")))]
                let quantized_values: candle_core::Result<Vec<_>> = {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(2)
                        .build()
                        .map_err(candle_core::Error::msg)?;

                    pool.install(|| {
                        use rayon::iter::IntoParallelRefIterator;
                        if silent {
                            tensors
                                .par_iter()
                                .enumerate()
                                .filter(|(_, (layer, _))| layer.isq_serde_supported())
                                .map(|(i, (layer, _))| {
                                    Ok((
                                        i.to_string(),
                                        match layer.serialize()? {
                                            Cow::Borrowed(_) => unreachable!(),
                                            Cow::Owned(owned) => owned,
                                        },
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
                                        match layer.serialize()? {
                                            Cow::Borrowed(_) => unreachable!(),
                                            Cow::Owned(owned) => owned,
                                        },
                                    ))
                                })
                                .collect::<candle_core::Result<Vec<_>>>()
                        }
                    })
                };

                let quantized_values = quantized_values?;

                let parent = serialized
                    .parent()
                    .context("Target UQFF path must have a filename!")?;

                std::fs::create_dir_all(parent)?;

                let file_stem = serialized
                    .file_stem()
                    .context("Target UQFF path must have a file stem!")?
                    .to_string_lossy()
                    .to_string();

                // Shard quantized values by cumulative byte size, max MAX_UQFF_SIZE_BYTES per file
                let mut current_chunk = Vec::new();
                let mut current_bytes: usize = 0;
                let mut shard_index = 0;

                // Every 10GB, flush the file. Then save any remaining tensors
                for (name, tensor) in quantized_values.iter() {
                    let tensor_bytes = tensor.len();
                    if !current_chunk.is_empty()
                        && current_bytes + tensor_bytes > MAX_UQFF_SIZE_BYTES
                    {
                        let mut shard_path = parent.to_path_buf();
                        shard_path.push(format!("{file_stem}-{shard_index}.uqff"));
                        info!(
                            "Writing shard {} to `{}`",
                            shard_index,
                            shard_path.display()
                        );
                        safetensors::serialize_to_file(current_chunk.clone(), None, &shard_path)?;
                        shard_index += 1;
                        current_chunk.clear();
                        current_bytes = 0;
                    }
                    current_bytes += tensor_bytes;
                    current_chunk.push((name, CowBytesView::new(Cow::Borrowed(tensor))));
                }

                if !current_chunk.is_empty() {
                    let mut shard_path = parent.to_path_buf();
                    shard_path.push(format!("{file_stem}-{shard_index}.uqff"));
                    info!(
                        "Writing final shard {} to `{}`",
                        shard_index,
                        shard_path.display()
                    );
                    safetensors::serialize_to_file(current_chunk.clone(), None, &shard_path)?;
                }

                let residual = match organization {
                    IsqOrganization::Default => self.residual_tensors(),
                    IsqOrganization::MoeExpertsOnly => self
                        .residual_tensors_moe_experts_only()
                        .unwrap_or(self.residual_tensors()),
                };

                let residual_out = parent.join(UQFF_RESIDUAL_SAFETENSORS);
                let config_out = parent.join("config.json");
                let modules_out = parent.join("modules.json");
                let tokenizer_out = parent.join("tokenizer.json");
                let tokenizer_cfg_out = parent.join("tokenizer_config.json");
                let chat_template_jinja_out = parent.join("chat_template.jinja");
                let gen_cfg_out = parent.join("generation_config.json");
                let processor_out = parent.join("processor_config.json");
                let preprocessor_out = parent.join("preprocessor_config.json");

                info!(
                    "Serializing {} residual tensors to `{}`.",
                    residual.len(),
                    residual_out.display()
                );

                safetensors::serialize_to_file(residual, None, &residual_out)?;

                let UqffFullSer {
                    tokenizer,
                    template_filename,
                    modules,
                    module_paths,
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
                    let template =
                        std::fs::read(template_filename).map_err(candle_core::Error::msg)?;

                    if template_filename.extension().map(|e| e.to_str()) == Some(Some("jinja")) {
                        info!(
                            "Serializing chat template to `{}`.",
                            chat_template_jinja_out.display()
                        );
                        std::fs::write(&chat_template_jinja_out, template)
                            .map_err(candle_core::Error::msg)?;
                    } else {
                        info!(
                            "Serializing tokenizer config to `{}`.",
                            tokenizer_cfg_out.display()
                        );
                        std::fs::write(&tokenizer_cfg_out, template)
                            .map_err(candle_core::Error::msg)?;
                    }
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

                if let Some(modules) = modules {
                    info!(
                        "Serializing modules manifest to `{}`.",
                        modules_out.display()
                    );

                    std::fs::write(&modules_out, modules).map_err(candle_core::Error::msg)?;

                    if let Some(module_paths) = module_paths {
                        for module in module_paths {
                            match module {
                                EmbeddingModulePaths::Transformer { path }
                                | EmbeddingModulePaths::Pooling { path, .. }
                                | EmbeddingModulePaths::Dense { path, .. }
                                | EmbeddingModulePaths::Normalize { path } => {
                                    if path.is_empty() {
                                        continue;
                                    }
                                    let module_dir = parent.join(path.as_str());
                                    std::fs::create_dir_all(&module_dir)
                                        .map_err(candle_core::Error::msg)?;

                                    match module {
                                        EmbeddingModulePaths::Pooling { config, .. } => {
                                            let dest = module_dir.join("config.json");
                                            if config != &dest {
                                                std::fs::copy(config, &dest)
                                                    .map_err(candle_core::Error::msg)?;
                                            }
                                        }
                                        EmbeddingModulePaths::Dense { config, model, .. } => {
                                            let dest_cfg = module_dir.join("config.json");
                                            if config != &dest_cfg {
                                                std::fs::copy(config, &dest_cfg)
                                                    .map_err(candle_core::Error::msg)?;
                                            }
                                            let dest_model = module_dir.join("model.safetensors");
                                            if model != &dest_model {
                                                std::fs::copy(model, &dest_model)
                                                    .map_err(candle_core::Error::msg)?;
                                            }
                                        }
                                        EmbeddingModulePaths::Transformer { .. }
                                        | EmbeddingModulePaths::Normalize { .. } => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn load_from_artifacts(
        &mut self,
        device: Device,
        topology: Option<&Topology>,
        silent: bool,
        artifacts: &[PathBuf],
    ) -> candle_core::Result<()> {
        let (tensors, mapper) = self.get_layers();
        let total_tensors = tensors.len();

        let layers = topology.map(|x| {
            x.layers
                .iter()
                .filter_map(|topo| topo.as_ref().map(|x| (x.isq, x.device.clone())))
                .collect::<Vec<_>>()
        });

        let mut devices = Vec::new();
        let mut comms = Vec::new();
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
            comms.push(mapper.get_comm_for(layer_num.unwrap_or(0))?)
        }

        let artifacts = unsafe { candle_core::safetensors::MmapedSafetensors::multi(artifacts)? };

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
        configure_progress_bar(&bar);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.red/magenta}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let t_start = Instant::now();

        let guard = QuantizeOntoGuard::new();

        if silent {
            (0..tensors.len())
                .into_par_iter()
                .zip(tensors)
                .map(|(i, (tensor, _))| {
                    if let Some(artifact) = artifact_isqs.get(&i) {
                        let artifact = artifact.data();

                        let comm = comms[i].clone();
                        let deserialized = match tensor.is_distributed() {
                            Some(DistributedKind::ColumnParallel) => {
                                ColumnParallelLayer::deserialize(
                                    Cow::from(artifact),
                                    &devices[i],
                                    &comm,
                                    guard.clone(),
                                )?
                            }
                            Some(DistributedKind::RowParallel) => RowParallelLayer::deserialize(
                                Cow::from(artifact),
                                &devices[i],
                                &comm,
                                guard.clone(),
                            )?,
                            Some(DistributedKind::Replicated) => ReplicatedLayer::deserialize(
                                Cow::from(artifact),
                                &devices[i],
                                &comm,
                                guard.clone(),
                            )?,
                            None => {
                                // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
                                let isq_type = artifact[mistralrs_quant::UQFF_QUANT_TYPE_OFFSET];
                                match QuantizedSerdeType::try_from(isq_type as usize)? {
                                    QuantizedSerdeType::Gguf => GgufMatMul::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Unquant => UnquantLinear::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Hqq => HqqLayer::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Fp8 => FP8Linear::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Afq => AfqLayer::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::F8Q8 => F8Q8Linear::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                }
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

                        let comm = comms[i].clone();
                        let deserialized = match tensor.is_distributed() {
                            Some(DistributedKind::ColumnParallel) => {
                                ColumnParallelLayer::deserialize(
                                    Cow::from(artifact),
                                    &devices[i],
                                    &comm,
                                    guard.clone(),
                                )?
                            }
                            Some(DistributedKind::RowParallel) => RowParallelLayer::deserialize(
                                Cow::from(artifact),
                                &devices[i],
                                &comm,
                                guard.clone(),
                            )?,
                            Some(DistributedKind::Replicated) => ReplicatedLayer::deserialize(
                                Cow::from(artifact),
                                &devices[i],
                                &comm,
                                guard.clone(),
                            )?,
                            None => {
                                // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
                                let isq_type = artifact[mistralrs_quant::UQFF_QUANT_TYPE_OFFSET];
                                match QuantizedSerdeType::try_from(isq_type as usize)? {
                                    QuantizedSerdeType::Gguf => GgufMatMul::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Unquant => UnquantLinear::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Hqq => HqqLayer::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Fp8 => FP8Linear::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::Afq => AfqLayer::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                    QuantizedSerdeType::F8Q8 => F8Q8Linear::deserialize(
                                        Cow::from(artifact),
                                        &devices[i],
                                        &comm,
                                        guard.clone(),
                                    )?,
                                }
                            }
                        };
                        *tensor = deserialized;
                    }
                    Ok(())
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
        }

        // Verify no DummyLayers remain after deserialization
        {
            let (check_tensors, _) = self.get_layers();
            for (i, (tensor, layer_num)) in check_tensors.iter().enumerate() {
                if tensor.name() == "dummy" {
                    candle_core::bail!(
                        "DummyLayer not replaced at index {i}, layer {layer_num:?} after load_from_artifacts"
                    );
                }
            }
        }

        let delta = Instant::now().duration_since(t_start).as_secs_f32();
        info!("Loaded in-situ quantization artifacts into {total_tensors} total tensors. Took {delta:.2}s", );

        Ok(())
    }
}

/// Trait for loading models with ISQ.
pub(crate) trait IsqModelLoader {
    /// Regex to match layers which will have standard *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

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
