use candle_core::{backend::BackendStorage, MetalStorage};

use super::*;
use crate::metal_kernels::{
    utils::{linear_split, Output},
    Kernels,
};

fn metal_storage(storage: &Storage) -> Result<&MetalStorage> {
    match storage {
        Storage::Metal(storage) => Ok(storage),
        _ => candle_core::bail!("LFM3 convolution expects Metal storage"),
    }
}

impl ConvInputs<'_> {
    pub(super) fn metal(&self) -> Result<(Tensor, Tensor)> {
        let (projected, projected_layout) = self.projected.storage_and_layout();
        let (weight, weight_layout) = self.weight.storage_and_layout();
        let (state, state_layout) = self.state.storage_and_layout();
        let bias = self.bias.map(Tensor::storage_and_layout);
        let projected = metal_storage(&projected)?;
        let weight = metal_storage(&weight)?;
        let state = metal_storage(&state)?;
        let (bias, bias_offset) = match &bias {
            Some((storage, layout)) => (metal_storage(storage)?, layout.start_offset()),
            None => (weight, weight_layout.start_offset()),
        };
        let dtype = self.projected.dtype();
        let type_name = match dtype {
            DType::F32 => "float",
            DType::F16 => "half",
            DType::BF16 => "bfloat",
            dtype => candle_core::bail!("Unsupported LFM3 convolution dtype {dtype:?}"),
        };
        let specialization = if self.width == OPTIMIZED_KERNEL_WIDTH {
            OPTIMIZED_KERNEL_WIDTH
        } else {
            0
        };
        let device = projected.device();
        let pipeline = Kernels::global()
            .load_pipeline(
                device.device(),
                format!("lfm3_conv_{type_name}_{specialization}"),
            )
            .map_err(candle_core::Error::wrap)?;
        let output = device.new_buffer(self.output_len(), dtype, "lfm3-conv-output")?;
        let next_state = device.new_buffer(self.state_len(), dtype, "lfm3-conv-state")?;
        let command_encoder = device.command_encoder()?;
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = command_encoder.as_ref();
        encoder.set_label("lfm3-gated-conv");
        encoder.set_compute_pipeline_state(&pipeline);
        let element_size = dtype.size_in_bytes();
        crate::set_params!(
            encoder,
            (
                (
                    projected.buffer(),
                    projected_layout.start_offset() * element_size
                ),
                (weight.buffer(), weight_layout.start_offset() * element_size),
                (bias.buffer(), bias_offset * element_size),
                (state.buffer(), state_layout.start_offset() * element_size),
                Output::new(&output),
                Output::new(&next_state),
                self.params.as_slice()
            )
        );
        let (grid, group) = linear_split(&pipeline, self.output_len());
        encoder.dispatch_thread_groups(grid, group);
        Ok(self.tensors(
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                self.output_len(),
                dtype,
            )),
            Storage::Metal(MetalStorage::new(
                next_state,
                device.clone(),
                self.state_len(),
                dtype,
            )),
        ))
    }
}
