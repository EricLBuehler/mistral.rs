use std::{env, path::PathBuf, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device};
use mistralrs_quant::{IsqType, QuantizedWeightSource, ShardedVarBuilder, UqffReader};
use tracing::{info, warn};

use super::{
    auto_device_map, AutoDeviceMapParams, AutoMapSizingInput, DeviceMappedModelLoader,
    ResourceSource,
};
use crate::{
    device_map::{self, DeviceMapper},
    distributed::{self, TensorParallelism, WorkerTransferData},
    matformer::{MatformerConfig, MatformerSliceConfig},
    paged_attention::ModelConfigLike,
    pipeline::{
        isq::sanitize_quantized_weight_source_config, HfConfigOverrides, IsqModelLoader, ModelPaths,
    },
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
    DeviceMapSetting, IsqOrganization, PagedAttentionConfig, Topology, TryIntoDType,
};

pub(crate) struct ModelConfigLayers<'a> {
    pub paths: &'a dyn ModelPaths,
    pub prepared: Option<&'a str>,
    pub from_uqff: bool,
    pub overrides: Option<&'a HfConfigOverrides>,
    pub mtp: bool,
}

/// Reads the model config and applies the UQFF, user override and MTP layers in that order.
pub(crate) fn layered_model_config(layers: ModelConfigLayers<'_>) -> Result<String> {
    let config = match layers.prepared {
        Some(config) => config.to_string(),
        None => super::load_model_config(layers.paths.get_config_filename(), !layers.from_uqff)?,
    };
    let config = if layers.from_uqff {
        sanitize_quantized_weight_source_config(&config)?
    } else {
        config
    };
    let config = match layers.overrides {
        Some(overrides) => overrides.apply(&config)?,
        None => config,
    };
    if layers.mtp {
        super::inject_mtp_config_flag(&config)
    } else {
        Ok(config)
    }
}

/// When a load is pinned to CUDA device 0 instead of spanning every similar device.
#[derive(Clone, Copy)]
pub(crate) enum SingleCudaDevice {
    NcclRequested,
    TensorParallel,
}

pub(crate) struct LoadDevices {
    pub tensor_parallelism: TensorParallelism,
    pub device: Device,
    pub available: Vec<Device>,
}

pub(crate) fn resolve_load_devices(
    model_config: &dyn ModelConfigLike,
    device: &Device,
    write_uqff: bool,
    single_cuda: SingleCudaDevice,
    paged: Option<&PagedAttentionConfig>,
) -> Result<LoadDevices> {
    let use_nccl = mistralrs_quant::distributed::use_nccl();
    let tensor_parallelism =
        distributed::resolve_tensor_parallelism(model_config, use_nccl, write_uqff)?;
    let use_distributed = tensor_parallelism.is_enabled();
    let pin_single_cuda = match single_cuda {
        SingleCudaDevice::NcclRequested => use_nccl && !write_uqff,
        SingleCudaDevice::TensorParallel => use_distributed,
    };

    let available = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
        let payload: WorkerTransferData = serde_json::from_str(&payload)?;
        let WorkerTransferData::Init { worker_rank, .. } = payload;
        vec![Device::new_cuda(worker_rank + 1)?]
    } else if pin_single_cuda {
        vec![Device::new_cuda(0)?]
    } else {
        device_map::get_all_similar_devices(device)?
    };
    if !write_uqff {
        if let Some(paged) = paged {
            paged.initialize_memory_budget(&available)?;
        }
    }
    #[cfg(feature = "cuda")]
    for device in &available {
        if let Device::Cuda(dev) = device {
            unsafe { dev.disable_event_tracking() };
        }
    }
    let device = if use_distributed {
        available[0].clone()
    } else {
        device.clone()
    };
    Ok(LoadDevices {
        tensor_parallelism,
        device,
        available,
    })
}

pub(crate) struct WeightSources {
    pub uqff_reader: Option<Arc<UqffReader>>,
    pub prepared: Option<Arc<dyn QuantizedWeightSource>>,
    /// The source weights are actually read from: UQFF wins over a prepared source.
    pub active: Option<Arc<dyn QuantizedWeightSource>>,
}

pub(crate) fn open_weight_sources(
    from_uqff: Option<&Vec<PathBuf>>,
    prepared_weights: Option<&ShardedVarBuilder>,
) -> Result<WeightSources> {
    let uqff_reader = from_uqff
        .map(|files| UqffReader::open(files).map(Arc::new))
        .transpose()?;
    let prepared = prepared_weights.and_then(|weights| weights.weight_source().cloned());
    let active = uqff_reader
        .clone()
        .map(|reader| reader as Arc<dyn QuantizedWeightSource>)
        .or(prepared.clone());
    Ok(WeightSources {
        uqff_reader,
        prepared,
        active,
    })
}

pub(crate) fn load_matformer_slice(
    config_path: Option<&PathBuf>,
    slice_name: Option<&String>,
) -> Result<Option<MatformerSliceConfig>> {
    let Some(path) = config_path else {
        return Ok(None);
    };
    info!("Loading Matformer config from {:?}", path);
    let config = Arc::new(MatformerConfig::from_file(path)?);
    let Some(slice_name) = slice_name else {
        warn!("Matformer config loaded but no slice name specified. Models will use their default slice.");
        return Ok(None);
    };
    info!("Using Matformer slice: {}", slice_name);
    Ok(Some(MatformerSliceConfig::new(slice_name.clone(), config)))
}

pub(crate) struct DeviceMapPlanRequest<'a> {
    pub setting: DeviceMapSetting,
    pub loader: &'a dyn DeviceMappedModelLoader,
    pub config: &'a str,
    pub dtype: &'a dyn TryIntoDType,
    pub write_uqff: bool,
    pub use_distributed: bool,
    pub available_devices: &'a [Device],
    pub paths: &'a dyn ModelPaths,
    pub prepared_weights: Option<&'a ShardedVarBuilder>,
    pub sources: &'a WeightSources,
    pub in_situ_quant: Option<IsqType>,
    pub topology: Option<&'a Topology>,
    pub organization: IsqOrganization,
    pub has_lora_adapters: bool,
    pub matformer: Option<&'a MatformerSliceConfig>,
    pub packed_non_mapped: bool,
    /// Replaces the user's automatic mapping params when the loader derives its own.
    pub auto_params: Option<&'a AutoDeviceMapParams>,
}

/// Turns the requested mapping into a concrete one, sizing the model when it is automatic.
pub(crate) fn plan_device_map(
    request: DeviceMapPlanRequest<'_>,
    paged: Option<&mut PagedAttentionConfig>,
) -> Result<DeviceMapSetting> {
    if request.write_uqff {
        return Ok(DeviceMapSetting::dummy());
    }
    if request.use_distributed {
        return Ok(DeviceMapSetting::DummyNccl {
            nm_device: request.available_devices[0].clone(),
        });
    }
    let DeviceMapSetting::Auto(user_params) = &request.setting else {
        return Ok(request.setting);
    };
    let params = request.auto_params.unwrap_or(user_params);
    let loader = request.loader;
    let config = request.config;

    let dtype = request
        .dtype
        .try_into_dtype(&request.available_devices.iter().collect::<Vec<_>>())?;
    let sizing_source = ResourceSource {
        weights: request.prepared_weights,
        paths: request.paths.get_weight_filenames(),
    };
    let checkpoint = sizing_source.checkpoint_tensors()?;

    let (mut layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
        super::auto_device_map_sizes(
            loader,
            AutoMapSizingInput {
                config,
                dtype,
                weight_files: request.paths.get_weight_filenames(),
                checkpoint: &checkpoint,
                uqff: request.sources.uqff_reader.is_some(),
                weight_source: request.sources.active.as_ref(),
                prepared_weight_source: request.sources.prepared.as_ref(),
                in_situ_quant: request.in_situ_quant,
                topology: request.topology,
                organization: request.organization,
                has_lora_adapters: request.has_lora_adapters,
                matformer: request.matformer,
                packed_non_mapped: request.packed_non_mapped,
            },
        )?;

    let resources = loader.layer_resources(config, sizing_source)?;
    let total_model_size_in_bytes = super::add_resource_bytes(
        &mut layer_sizes_in_bytes,
        total_model_size_in_bytes,
        &resources,
    )?;
    let layers = auto_device_map::get_device_layers(
        loader,
        config,
        loader.num_layers(config)?,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        request.available_devices,
        dtype,
        params,
        paged,
        &resources,
    )?;
    Ok(DeviceMapSetting::Map(layers))
}

pub(crate) struct DeviceMappingRequest<'a> {
    pub setting: &'a DeviceMapSetting,
    pub num_layers: usize,
    pub device: &'a Device,
    pub topology: Option<&'a Topology>,
    pub available_devices: &'a [Device],
    pub dtype: &'a dyn TryIntoDType,
    pub write_uqff: bool,
}

pub(crate) struct ResolvedDeviceMapping {
    /// Kept by the pipeline; `mapper` is consumed by the model load.
    pub pipeline_mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub layer_devices: Vec<Option<Device>>,
    pub dtype: DType,
}

pub(crate) fn resolve_device_mapping(
    request: DeviceMappingRequest<'_>,
) -> Result<ResolvedDeviceMapping> {
    let DeviceMappingRequest {
        setting,
        num_layers,
        device,
        topology,
        available_devices,
        dtype,
        write_uqff,
    } = request;
    let cpu = Device::Cpu;
    let (mapper_device, topology) = if write_uqff {
        (&cpu, None)
    } else {
        (device, topology)
    };
    let pipeline_mapper =
        setting.into_mapper(num_layers, mapper_device, topology, available_devices)?;
    let mapper = setting.into_mapper(num_layers, mapper_device, topology, available_devices)?;
    let layer_devices = (0..num_layers)
        .map(|layer| mapper.device_for(layer, false).cloned())
        .collect();
    let dtype = crate::pipeline::isq_flow::resolve_weight_load_dtype(
        dtype,
        mapper.as_ref(),
        available_devices,
        write_uqff,
    )?;
    Ok(ResolvedDeviceMapping {
        pipeline_mapper,
        mapper,
        layer_devices,
        dtype,
    })
}

pub(crate) struct MappedWeightsRequest<'a> {
    pub paths: &'a dyn ModelPaths,
    pub dtype: DType,
    pub load_device: &'a Device,
    pub layer_devices: Vec<Option<Device>>,
    pub silent: bool,
    pub config: &'a str,
    /// UQFF supplies the quantized layers, so their safetensors entries load as placeholders.
    pub dummy_isq_layers: bool,
    pub moe_experts_only: bool,
    pub device_for_tensor: Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync>,
    pub uqff_reader: Option<Arc<UqffReader>>,
}

/// Opens the checkpoint in `paths` as a device-mapped builder.
pub(crate) fn open_mapped_weights(
    loader: &dyn IsqModelLoader,
    request: MappedWeightsRequest<'_>,
) -> Result<ShardedVarBuilder> {
    let dummy_regexes = if request.dummy_isq_layers {
        Some(Arc::new(if request.moe_experts_only {
            loader.isq_layer_regexes_moqe(request.config)?
        } else {
            loader.isq_layer_regexes(request.config)?
        }))
    } else {
        None
    };
    let vb = from_mmaped_safetensors(
        request.paths.get_weight_filenames().to_vec(),
        Vec::new(),
        Some(request.dtype),
        request.load_device,
        request.layer_devices,
        request.silent,
        dummy_regexes,
        |_| true,
        request.device_for_tensor,
    )?;
    Ok(match request.uqff_reader {
        Some(reader) => vb.with_uqff_reader(reader),
        None => vb,
    })
}
