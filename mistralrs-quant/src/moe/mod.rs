//! MoE support that is independent of any specific quant scheme.

pub mod cuda;
#[cfg(has_cutlass_moe_kernels)]
pub mod cutlass;

/// CUTLASS grouped-GEMM MoE forward; errors when the kernels were not compiled in.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_fused_moe(
    xs: &candle_core::Tensor,
    gate_up: &candle_core::Tensor,
    down: &candle_core::Tensor,
    topk_ids: &candle_core::Tensor,
    topk_weights: &candle_core::Tensor,
    num_experts: usize,
    act: cuda::GatedAct,
    dev: &candle_core::CudaDevice,
) -> candle_core::Result<candle_core::Tensor> {
    #[cfg(has_cutlass_moe_kernels)]
    {
        cutlass::cutlass_fused_moe(
            xs,
            gate_up,
            down,
            topk_ids,
            topk_weights,
            num_experts,
            act,
            dev,
        )
    }
    #[cfg(not(has_cutlass_moe_kernels))]
    {
        let _ = (
            xs,
            gate_up,
            down,
            topk_ids,
            topk_weights,
            num_experts,
            act,
            dev,
        );
        candle_core::bail!("CUTLASS MoE kernels were not compiled in (requires sm_80+ at build)")
    }
}

/// Whether the CUTLASS grouped-GEMM MoE kernels were compiled in and the device can run them.
pub fn cutlass_moe_available(dev: &candle_core::CudaDevice) -> bool {
    #[cfg(has_cutlass_moe_kernels)]
    {
        use candle_core::cuda::cudarc::driver::{result, sys};
        let cu_device = dev.cuda_stream().context().cu_device();
        let major = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        }
        .unwrap_or(0);
        major >= 8
    }
    #[cfg(not(has_cutlass_moe_kernels))]
    {
        let _ = dev;
        false
    }
}
