use std::{fmt::Debug, sync::Arc};

use crate::TryIntoDType;
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use super::peer::CudaPeerAccess;

pub trait DeviceMapper: Debug {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;

    fn set_device(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder;
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device>;
    fn get_unique_devices(&self) -> Vec<Device>;
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor>;
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder;
    fn num_device_mapping_layers(&self) -> usize;
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>>;

    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType>;
}

#[derive(Debug)]
pub struct LayerDeviceMapper {
    mappings: Vec<Device>,
    nm_device: Device,
    cuda_peer_access: CudaPeerAccess,
}

impl LayerDeviceMapper {
    pub(super) fn new(
        mappings: Vec<Device>,
        nm_device: Device,
        cuda_peer_access: CudaPeerAccess,
    ) -> Self {
        Self {
            mappings,
            nm_device,
            cuda_peer_access,
        }
    }
}

impl DeviceMapper for LayerDeviceMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        self.cuda_peer_access
            .to_device(&input, &self.mappings[layer])
    }
    fn set_device(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            return varbuilder;
        }
        varbuilder.set_device(self.mappings[layer].clone())
    }
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        self.mappings.get(layer)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.mappings.iter().fold(Vec::new(), |mut acc, device| {
            if !acc.iter().any(|d| d.same_device(device)) {
                acc.push(device.clone());
            }
            acc
        })
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            self.cuda_peer_access.to_device(x, &self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.mappings.iter().collect::<Vec<_>>())
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.mappings.len()
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
pub struct DummyDeviceMapper {
    pub(crate) nm_device: Device,
}

impl DummyDeviceMapper {
    pub(super) fn new(nm_device: Device) -> Self {
        Self { nm_device }
    }
}

impl DeviceMapper for DummyDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.nm_device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        vec![self.nm_device.clone()]
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&[&self.nm_device])
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        1
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
pub struct NcclDeviceMapper {
    nm_device: Device,
    model_layers: usize,
    comm: Option<Arc<mistralrs_quant::Comm>>,
}

impl NcclDeviceMapper {
    pub(super) fn new(
        nm_device: Device,
        model_layers: usize,
        comm: Option<Arc<mistralrs_quant::Comm>>,
    ) -> Self {
        Self {
            nm_device,
            model_layers,
            comm,
        }
    }
}

impl DeviceMapper for NcclDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.nm_device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        vec![self.nm_device.clone()]
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&[&self.nm_device])
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.model_layers
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        if let Some(comm) = &self.comm {
            Ok(comm.clone())
        } else {
            let id = mistralrs_quant::Id::new();
            Ok(Arc::new(mistralrs_quant::Comm::from_device(
                id,
                self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
                0,
                1,
            )?))
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct NcclPipelineParallelMapper {
    mappings: Vec<(Arc<mistralrs_quant::Comm>, Device)>,
    nm_device: Device,
    cuda_peer_access: CudaPeerAccess,
}

impl DeviceMapper for NcclPipelineParallelMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        self.cuda_peer_access
            .to_device(&input, &self.mappings[layer].1)
    }
    fn set_device(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            return varbuilder;
        }
        varbuilder.set_device(self.mappings[layer].1.clone())
    }
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        self.mappings.get(layer).map(|(_, x)| x)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.mappings
            .iter()
            .fold(Vec::new(), |mut acc, (_, device)| {
                if !acc.iter().any(|d| d.same_device(device)) {
                    acc.push(device.clone());
                }
                acc
            })
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            self.cuda_peer_access.to_device(x, &self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.mappings.iter().map(|(_, x)| x).collect::<Vec<_>>())
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.mappings.len()
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        Ok(self.mappings[layer_idx].0.clone())
    }
}
