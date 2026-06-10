//! CUTLASS 2.x grouped-GEMM MoE fallback (kernels in `kernels/cutlass_moe/*.cu`).
//! Universal bf16 path for sm_80+: expert-sorted grouped GEMMs with device-resident problem
//! sizes, so the forward never syncs to host.

#![cfg(has_cutlass_moe_kernels)]

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{CudaDevice, DType, Result, Storage, Tensor};
use half::bf16;

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

mod ffi {
    use candle_core::cuda::cudarc::driver::sys::CUstream;
    use core::ffi::c_void;

    extern "C" {
        pub fn launch_cutlass_moe_problem_sizes(
            topk_ids: *const i32,
            problem_sizes1: *mut i32,
            problem_sizes2: *mut i32,
            atomic_buffer: *mut i32,
            num_experts: i32,
            topk_length: i32,
            n: i32,
            k: i32,
            is_gated: bool,
            stream: CUstream,
        );
        pub fn launch_cutlass_moe_expert_offsets(
            problem_sizes1: *const i32,
            expert_offsets: *mut i32,
            atomic_buffer: *mut i32,
            num_experts: i32,
            stream: CUstream,
        );
        pub fn launch_cutlass_moe_arg_sorts(
            topk_ids: *const i32,
            input_permutation: *mut i32,
            output_permutation: *mut i32,
            atomic_buffer: *mut i32,
            num_experts: i32,
            topk_length: i32,
            topk: i32,
            stream: CUstream,
        );
        pub fn launch_cutlass_moe_gather_rows_bf16(
            dst: *mut c_void,
            src: *const c_void,
            map: *const i32,
            num_rows: i32,
            k: i32,
            stream: CUstream,
        );
        pub fn launch_cutlass_moe_gather_weighted_bf16(
            out: *mut c_void,
            input: *const c_void,
            out_perm: *const i32,
            weights: *const f32,
            num_rows: i32,
            k: i32,
            stream: CUstream,
        );
        pub fn launch_cutlass_moe_group_starts_bf16(
            expert_offsets: *const i32,
            a_base: *const c_void,
            b_base: *const c_void,
            d_base: *mut c_void,
            a_ptrs: *const *const c_void,
            b_ptrs: *const *const c_void,
            d_ptrs: *mut *mut c_void,
            lda: *mut i64,
            ldb: *mut i64,
            ldd: *mut i64,
            num_experts: i32,
            n: i64,
            k: i64,
            stream: CUstream,
        );
        pub fn launch_cutlass_moe_grouped_gemm_2x_bf16(
            a_ptrs: *const *const c_void,
            b_ptrs: *const *const c_void,
            d_ptrs: *mut *mut c_void,
            problem_sizes: *const i32,
            problem_count: i32,
            lda: *mut i64,
            ldb: *mut i64,
            ldd: *mut i64,
            workspace: *mut c_void,
            workspace_size: usize,
            tile_cfg: i32,
            stream: CUstream,
        ) -> i32;
        pub fn cutlass_moe_grouped_gemm_2x_workspace_size(problem_count: i32) -> usize;
    }
}

fn bf16_cuda_slice(t: &Tensor, what: &str) -> Result<(std::sync::Arc<CudaSlice<bf16>>, usize)> {
    let (storage, layout) = t.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?.clone(),
        _ => candle_core::bail!("{what} must be a cuda tensor"),
    };
    Ok((std::sync::Arc::new(slice), layout.start_offset()))
}

/// Fused bf16 MoE forward via two CUTLASS grouped GEMMs.
///
/// `xs`: [num_tokens, hidden], `gate_up`: [E, 2*inter, hidden], `down`: [E, hidden, inter]
/// (ENK layout), `topk_ids`: [num_tokens, topk] u32, `topk_weights`: [num_tokens, topk].
/// Returns [num_tokens, hidden].
#[allow(clippy::too_many_arguments)]
pub fn cutlass_fused_moe(
    xs: &Tensor,
    gate_up: &Tensor,
    down: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    num_experts: usize,
    act: super::cuda::GatedAct,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let (num_tokens, hidden) = xs.dims2()?;
    let (_, two_inter, k1) = gate_up.dims3()?;
    let inter = two_inter / 2;
    assert_eq!(k1, hidden, "gate_up K must equal hidden");
    assert_eq!(xs.dtype(), DType::BF16, "cutlass moe path is bf16-only");
    let topk = topk_ids.dim(1)?;
    let num_valid = num_tokens * topk;

    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();

    // Tile config by average rows per expert group; known host-side, no sync.
    let avg_m = num_valid / num_experts.max(1);
    let tile_cfg: i32 = if avg_m >= 96 {
        0
    } else if avg_m >= 48 {
        1
    } else {
        2
    };

    let ti_flat = topk_ids.flatten_all()?.contiguous()?;
    let (ti_storage, ti_layout) = ti_flat.storage_and_layout();
    let ti_slice = match &*ti_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("topk_ids must be a cuda tensor"),
    };
    assert_eq!(ti_layout.start_offset(), 0, "expected contiguous topk_ids");

    let tw_flat = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (tw_storage, tw_layout) = tw_flat.storage_and_layout();
    let tw_slice = match &*tw_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("topk_weights must be a cuda tensor"),
    };
    assert_eq!(
        tw_layout.start_offset(),
        0,
        "expected contiguous topk_weights"
    );

    let (xs_slice, xs_off) = bf16_cuda_slice(xs, "xs")?;
    let (gu_slice, gu_off) = bf16_cuda_slice(gate_up, "gate_up")?;
    let (dn_slice, dn_off) = bf16_cuda_slice(down, "down")?;

    let mut ps1 = unsafe { dev.alloc::<i32>(num_experts * 3)? };
    let mut ps2 = unsafe { dev.alloc::<i32>(num_experts * 3)? };
    let mut atomic = unsafe { dev.alloc::<i32>(num_experts)? };
    let mut offsets = unsafe { dev.alloc::<i32>(num_experts + 1)? };
    let mut in_perm = unsafe { dev.alloc::<i32>(num_valid)? };
    let mut out_perm = unsafe { dev.alloc::<i32>(num_valid)? };
    let mut a_perm = unsafe { dev.alloc::<bf16>(num_valid * hidden)? };
    let mut c1 = unsafe { dev.alloc::<bf16>(num_valid * two_inter)? };
    let mut c2 = unsafe { dev.alloc::<bf16>(num_valid * hidden)? };
    let mut out = unsafe { dev.alloc::<bf16>(num_valid * hidden)? };
    let mut a_ptrs = unsafe { dev.alloc::<u64>(num_experts)? };
    let mut b_ptrs = unsafe { dev.alloc::<u64>(num_experts)? };
    let mut d_ptrs = unsafe { dev.alloc::<u64>(num_experts)? };
    let mut lda = unsafe { dev.alloc::<i64>(num_experts)? };
    let mut ldb = unsafe { dev.alloc::<i64>(num_experts)? };
    let mut ldd = unsafe { dev.alloc::<i64>(num_experts)? };
    let ws_size = unsafe { ffi::cutlass_moe_grouped_gemm_2x_workspace_size(num_experts as i32) };
    let mut workspace = unsafe { dev.alloc::<u8>(ws_size.max(1))? };

    let (ti_ptr, _ti_g) = slice_ptr_on_stream(ti_slice, 0, &stream);
    let (tw_ptr, _tw_g) = slice_ptr_on_stream(tw_slice, 0, &stream);
    let (xs_ptr, _xs_g) = slice_ptr_on_stream(&xs_slice, xs_off, &stream);
    let (gu_ptr, _gu_g) = slice_ptr_on_stream(&gu_slice, gu_off, &stream);
    let (dn_ptr, _dn_g) = slice_ptr_on_stream(&dn_slice, dn_off, &stream);
    let (ps1_ptr, ps1_g) = slice_ptr_mut_on_stream(&mut ps1, 0, &stream);
    let (ps2_ptr, ps2_g) = slice_ptr_mut_on_stream(&mut ps2, 0, &stream);
    let (at_ptr, at_g) = slice_ptr_mut_on_stream(&mut atomic, 0, &stream);
    let (off_ptr, off_g) = slice_ptr_mut_on_stream(&mut offsets, 0, &stream);
    let (ip_ptr, ip_g) = slice_ptr_mut_on_stream(&mut in_perm, 0, &stream);
    let (op_ptr, op_g) = slice_ptr_mut_on_stream(&mut out_perm, 0, &stream);
    let (ap_ptr, ap_g) = slice_ptr_mut_on_stream(&mut a_perm, 0, &stream);
    let (c1_ptr, c1_g) = slice_ptr_mut_on_stream(&mut c1, 0, &stream);
    let (c2_ptr, c2_g) = slice_ptr_mut_on_stream(&mut c2, 0, &stream);
    let (out_ptr, out_g) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    let (apt_ptr, apt_g) = slice_ptr_mut_on_stream(&mut a_ptrs, 0, &stream);
    let (bpt_ptr, bpt_g) = slice_ptr_mut_on_stream(&mut b_ptrs, 0, &stream);
    let (dpt_ptr, dpt_g) = slice_ptr_mut_on_stream(&mut d_ptrs, 0, &stream);
    let (lda_ptr, lda_g) = slice_ptr_mut_on_stream(&mut lda, 0, &stream);
    let (ldb_ptr, ldb_g) = slice_ptr_mut_on_stream(&mut ldb, 0, &stream);
    let (ldd_ptr, ldd_g) = slice_ptr_mut_on_stream(&mut ldd, 0, &stream);
    let (ws_ptr, ws_g) = slice_ptr_mut_on_stream(&mut workspace, 0, &stream);

    unsafe {
        ffi::launch_cutlass_moe_problem_sizes(
            ti_ptr as *const i32,
            ps1_ptr as *mut i32,
            ps2_ptr as *mut i32,
            at_ptr as *mut i32,
            num_experts as i32,
            num_valid as i32,
            inter as i32,
            hidden as i32,
            true,
            cu_stream,
        );
        ffi::launch_cutlass_moe_expert_offsets(
            ps1_ptr as *const i32,
            off_ptr as *mut i32,
            at_ptr as *mut i32,
            num_experts as i32,
            cu_stream,
        );
        ffi::launch_cutlass_moe_arg_sorts(
            ti_ptr as *const i32,
            ip_ptr as *mut i32,
            op_ptr as *mut i32,
            at_ptr as *mut i32,
            num_experts as i32,
            num_valid as i32,
            topk as i32,
            cu_stream,
        );
        ffi::launch_cutlass_moe_gather_rows_bf16(
            ap_ptr as *mut core::ffi::c_void,
            xs_ptr as *const core::ffi::c_void,
            ip_ptr as *const i32,
            num_valid as i32,
            hidden as i32,
            cu_stream,
        );
        ffi::launch_cutlass_moe_group_starts_bf16(
            off_ptr as *const i32,
            ap_ptr as *const core::ffi::c_void,
            gu_ptr as *const core::ffi::c_void,
            c1_ptr as *mut core::ffi::c_void,
            apt_ptr as *const *const core::ffi::c_void,
            bpt_ptr as *const *const core::ffi::c_void,
            dpt_ptr as *mut *mut core::ffi::c_void,
            lda_ptr as *mut i64,
            ldb_ptr as *mut i64,
            ldd_ptr as *mut i64,
            num_experts as i32,
            two_inter as i64,
            hidden as i64,
            cu_stream,
        );
        let status = ffi::launch_cutlass_moe_grouped_gemm_2x_bf16(
            apt_ptr as *const *const core::ffi::c_void,
            bpt_ptr as *const *const core::ffi::c_void,
            dpt_ptr as *mut *mut core::ffi::c_void,
            ps1_ptr as *const i32,
            num_experts as i32,
            lda_ptr as *mut i64,
            ldb_ptr as *mut i64,
            ldd_ptr as *mut i64,
            ws_ptr as *mut core::ffi::c_void,
            ws_size,
            tile_cfg,
            cu_stream,
        );
        if status != 0 {
            candle_core::bail!("cutlass grouped gemm 1 failed with status {status}");
        }
    }
    drop((ps1_g, at_g, ip_g, ap_g));

    drop(c1_g);
    let c1_storage = candle_core::CudaStorage::wrap_cuda_slice(c1, dev.clone());
    let c1_tensor = Tensor::from((Storage::Cuda(c1_storage), (num_valid, two_inter)));
    let act = super::cuda::act_and_mul(&c1_tensor, inter, act, dev)?;
    let (act_slice, act_off) = bf16_cuda_slice(&act, "act")?;
    let (act_ptr, _act_g) = slice_ptr_on_stream(&act_slice, act_off, &stream);

    unsafe {
        ffi::launch_cutlass_moe_group_starts_bf16(
            off_ptr as *const i32,
            act_ptr as *const core::ffi::c_void,
            dn_ptr as *const core::ffi::c_void,
            c2_ptr as *mut core::ffi::c_void,
            apt_ptr as *const *const core::ffi::c_void,
            bpt_ptr as *const *const core::ffi::c_void,
            dpt_ptr as *mut *mut core::ffi::c_void,
            lda_ptr as *mut i64,
            ldb_ptr as *mut i64,
            ldd_ptr as *mut i64,
            num_experts as i32,
            hidden as i64,
            inter as i64,
            cu_stream,
        );
        let status = ffi::launch_cutlass_moe_grouped_gemm_2x_bf16(
            apt_ptr as *const *const core::ffi::c_void,
            bpt_ptr as *const *const core::ffi::c_void,
            dpt_ptr as *mut *mut core::ffi::c_void,
            ps2_ptr as *const i32,
            num_experts as i32,
            lda_ptr as *mut i64,
            ldb_ptr as *mut i64,
            ldd_ptr as *mut i64,
            ws_ptr as *mut core::ffi::c_void,
            ws_size,
            tile_cfg,
            cu_stream,
        );
        if status != 0 {
            candle_core::bail!("cutlass grouped gemm 2 failed with status {status}");
        }
        ffi::launch_cutlass_moe_gather_weighted_bf16(
            out_ptr as *mut core::ffi::c_void,
            c2_ptr as *const core::ffi::c_void,
            op_ptr as *const i32,
            tw_ptr as *const f32,
            num_valid as i32,
            hidden as i32,
            cu_stream,
        );
    }
    drop((
        ps2_g, off_g, op_g, c2_g, out_g, apt_g, bpt_g, dpt_g, lda_g, ldb_g, ldd_g, ws_g,
    ));

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    let out_tensor = Tensor::from((Storage::Cuda(out_storage), (num_valid, hidden)));
    super::cuda::moe_sum_bf16(&out_tensor, num_tokens, topk, dev)
}
