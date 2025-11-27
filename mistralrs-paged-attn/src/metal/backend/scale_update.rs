use candle_core::{backend::BackendStorage, DType, Result, Storage, Tensor};

use crate::metal::kernels::{self, PagedAttentionDType};

#[derive(Debug, Clone)]
struct KvScaleUpdate {
    k_scales: Tensor,
    v_scales: Tensor,
}

impl candle_core::InplaceOp2 for KvScaleUpdate {
    fn name(&self) -> &'static str {
        "kvscale-update"
    }

    fn cpu_fwd(
        &self,
        _: &mut candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<()> {
        candle_core::bail!("kvscale-update is not implemented on CPU!")
    }

    fn metal_fwd(
        &self,
        k: &mut candle_core::MetalStorage,
        k_layout: &candle_core::Layout,
        v: &candle_core::MetalStorage,
        _: &candle_core::Layout,
    ) -> Result<()> {
        let ty = match k.dtype() {
            DType::F16 => PagedAttentionDType::F16,
            DType::BF16 => PagedAttentionDType::BF16,
            DType::F32 => PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported for kv_scale_update"),
        };

        let dev = k.device();
        let elem_count = k_layout.shape().elem_count();

        let (k_scales_storage, _) = self.k_scales.storage_and_layout();
        let k_scales = match &*k_scales_storage {
            Storage::Metal(m) => m,
            _ => candle_core::bail!("k_scales must be a metal tensor"),
        };

        let (v_scales_storage, _) = self.v_scales.storage_and_layout();
        let v_scales = match &*v_scales_storage {
            Storage::Metal(m) => m,
            _ => candle_core::bail!("v_scales must be a metal tensor"),
        };

        let encoder = dev.command_encoder()?;
        encoder.set_label("kv-scale-update");

        kernels::call_kv_scale_update(
            dev.device(),
            &encoder,
            &kernels::Kernels::new(),
            ty,
            k.buffer(),
            k_layout.start_offset() * k.dtype().size_in_bytes(),
            v.buffer(),
            0, // v_layout already incorporated by caller
            k_scales.buffer(),
            v_scales.buffer(),
            elem_count as i64,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

pub fn kv_scale_update(
    key: &Tensor,
    value: &Tensor,
    k_scales: &Tensor,
    v_scales: &Tensor,
) -> Result<()> {
    let op = KvScaleUpdate {
        k_scales: k_scales.to_owned(),
        v_scales: v_scales.to_owned(),
    };
    key.inplace_op2(value, &op)
}
