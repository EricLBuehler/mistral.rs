//! Chunked gated delta rule prefill on tensor cores: a parallel WY pass, a sequential per-head state
//! pass, and a parallel output pass, all in 64-token chunks with K = V = 128.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use std::sync::Arc;

use candle_core::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
use cutile::core::bf16 as tile_bf16;
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::bf16;

use super::warmup::CutileKernel;
use super::{catch_cutile_panic, context, jit_available};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

pub const GDN_PREFILL_CHUNK: usize = 64;
pub const GDN_PREFILL_HEAD_DIM: usize = 128;
const KV_SIZE: usize = GDN_PREFILL_HEAD_DIM * GDN_PREFILL_HEAD_DIM;

// Ragged counts travel as f32: cuTile keys its JIT cache on the divisibility of every i32 scalar and
// tensor dim, so with chunk-padded operands one compiled variant serves every sequence length.
#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::cutile_compiler;

    // w = A_inv (beta e^gcum k), u = A_inv (beta v) per (head, chunk), A_inv = (I + A)^-1 solved like
    // FLA's solve_tril: forward substitution on the 16x16 diagonal blocks (compact [C, 16] rows), then
    // the exact (I - N)(I + N^2) X_bd with N = X_bd A_off (N^4 = 0). tf32 tensor cores throughout:
    // f32 mmaf lowers to CUDA cores (seconds per prefill) and a Neumann product in A explodes.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn gdn_wy<const C: i32, const K: i32, const V: i32>(
        w_ptr: *mut bf16,
        u_ptr: *mut bf16,
        k: &Tensor<bf16, { [-1, -1, -1] }>,
        v: &Tensor<bf16, { [-1, -1, -1] }>,
        g: &Tensor<f32, { [-1, -1] }>,
        beta: &Tensor<f32, { [-1, -1] }>,
        num_chunks_f: f32,
        chunk: i32,
        k_dim: i32,
        v_dim: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let idx: i32 = pid.0;
        let num_chunks_t: Tile<f32, { [1] }> = broadcast_scalar(num_chunks_f, const_shape![1]);
        let num_chunks_i: Tile<i32, { [1] }> = convert_tile(num_chunks_t);
        let num_chunks_s: Tile<i32, { [] }> = num_chunks_i.reshape(const_shape![]);
        let num_chunks: i32 = tile_to_scalar(num_chunks_s);
        let bh: i32 = idx / num_chunks;
        let c: i32 = idx - bh * num_chunks;
        let pk = k.partition(const_shape![1, C, K]);
        let pv = v.partition(const_shape![1, C, V]);
        let pg = g.partition(const_shape![1, C]);
        let pb = beta.partition(const_shape![1, C]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let iota_c: Tile<i32, { [C] }> = iota(const_shape![C]);
        let ii_c1: Tile<i32, { [C, 1] }> = iota_c.reshape(const_shape![C, 1]);
        let ii: Tile<i32, { [C, C] }> = ii_c1.broadcast(const_shape![C, C]);
        let jj_1c: Tile<i32, { [1, C] }> = iota_c.reshape(const_shape![1, C]);
        let jj: Tile<i32, { [C, C] }> = jj_1c.broadcast(const_shape![C, C]);
        let strict: Tile<bool, { [C, C] }> = lt_tile(jj, ii);
        let diag: Tile<bool, { [C, C] }> = eq_tile(ii, jj);
        let zero_cc: Tile<f32, { [C, C] }> = constant(0.0f32, const_shape![C, C]);
        let one_cc: Tile<f32, { [C, C] }> = constant(1.0f32, const_shape![C, C]);
        let eye: Tile<f32, { [C, C] }> = select(diag, one_cc, zero_cc);
        let zero_cv: Tile<f32, { [C, V] }> = constant(0.0f32, const_shape![C, V]);
        let zero_ck: Tile<f32, { [C, K] }> = constant(0.0f32, const_shape![C, K]);

        let kc3: Tile<bf16, { [1, C, K] }> = pk.load([bh, c, 0]);
        let kc_b: Tile<bf16, { [C, K] }> = kc3.reshape(const_shape![C, K]);
        let vc3: Tile<bf16, { [1, C, V] }> = pv.load([bh, c, 0]);
        let vc_b: Tile<bf16, { [C, V] }> = vc3.reshape(const_shape![C, V]);
        let kc: Tile<f32, { [C, K] }> = convert_tile(kc_b);
        let vc: Tile<f32, { [C, V] }> = convert_tile(vc_b);
        let g_row: Tile<f32, { [1, C] }> = pg.load([bh, c]);
        let b_row: Tile<f32, { [1, C] }> = pb.load([bh, c]);
        let g1: Tile<f32, { [C] }> = g_row.reshape(const_shape![C]);
        let beta_col: Tile<f32, { [C, 1] }> = b_row.reshape(const_shape![C, 1]);
        let gcum: Tile<f32, { [C] }> = scan(g1, 0i32, reverse::Forward, 0.0f32, |acc, x| acc + x);
        let gcol: Tile<f32, { [C, 1] }> = gcum.reshape(const_shape![C, 1]);
        let grow: Tile<f32, { [1, C] }> = gcum.reshape(const_shape![1, C]);
        let gcol_cc: Tile<f32, { [C, C] }> = gcol.broadcast(const_shape![C, C]);
        let grow_cc: Tile<f32, { [C, C] }> = grow.broadcast(const_shape![C, C]);
        let gdiff: Tile<f32, { [C, C] }> = gcol_cc - grow_cc;
        let dfull: Tile<f32, { [C, C] }> = exp(gdiff);
        let d_strict: Tile<f32, { [C, C] }> = select(strict, dfull, zero_cc);
        let kt_b: Tile<bf16, { [K, C] }> = permute(kc_b, transpose);
        let kk: Tile<f32, { [C, C] }> = mmaf(kc_b, kt_b, zero_cc);
        let beta_cc: Tile<f32, { [C, C] }> = beta_col.broadcast(const_shape![C, C]);
        let a_mat: Tile<f32, { [C, C] }> = kk * d_strict * beta_cc;
        let blk_cc: Tile<i32, { [C, C] }> = broadcast_scalar(16i32, const_shape![C, C]);
        let bi: Tile<i32, { [C, C] }> = ii / blk_cc;
        let bj: Tile<i32, { [C, C] }> = jj / blk_cc;
        let same_block: Tile<bool, { [C, C] }> = eq_tile(bi, bj);
        let a_bd: Tile<f32, { [C, C] }> = select(same_block, a_mat, zero_cc);
        let a_off: Tile<f32, { [C, C] }> = a_mat - a_bd;
        let local_row: Tile<i32, { [C, C] }> = ii - bi * blk_cc;
        let zero_cb: Tile<f32, { [C, 16] }> = constant(0.0f32, const_shape![C, 16]);
        let one_cb: Tile<f32, { [C, 16] }> = constant(1.0f32, const_shape![C, 16]);
        let iota_b: Tile<i32, { [16] }> = iota(const_shape![16]);
        let bcols_cb: Tile<i32, { [C, 16] }> = iota_b
            .reshape(const_shape![1, 16])
            .broadcast(const_shape![C, 16]);
        let blk_c1: Tile<i32, { [C, 1] }> = broadcast_scalar(16i32, const_shape![C, 1]);
        let local_c1: Tile<i32, { [C, 1] }> = ii_c1 - (ii_c1 / blk_c1) * blk_c1;
        let local_cb: Tile<i32, { [C, 16] }> = local_c1.broadcast(const_shape![C, 16]);
        let diag_cb: Tile<bool, { [C, 16] }> = eq_tile(local_cb, bcols_cb);
        let mut x_c: Tile<f32, { [C, 16] }> = select(diag_cb, one_cb, zero_cb);
        for step in 1i32..16 {
            let step_cc: Tile<i32, { [C, C] }> = broadcast_scalar(step, const_shape![C, C]);
            let rows: Tile<bool, { [C, C] }> = eq_tile(local_row, step_cc);
            let a_rows: Tile<f32, { [C, C] }> = select(rows, a_bd, zero_cc);
            let a_rows_m: Tile<tf32, { [C, C] }> = convert_tile(a_rows);
            let x_c_m: Tile<tf32, { [C, 16] }> = convert_tile(x_c);
            let upd: Tile<f32, { [C, 16] }> = mmaf(a_rows_m, x_c_m, zero_cb);
            let x_next: Tile<f32, { [C, 16] }> = x_c - upd;
            x_c = x_next;
        }
        // expand the compact solve: X_bd = same_block ? X_c P : 0 with P[l, c] = (c mod 16 == l)
        let zero_bc: Tile<f32, { [16, C] }> = constant(0.0f32, const_shape![16, C]);
        let one_bc: Tile<f32, { [16, C] }> = constant(1.0f32, const_shape![16, C]);
        let blk_1c: Tile<i32, { [1, C] }> = broadcast_scalar(16i32, const_shape![1, C]);
        let jmod_1c: Tile<i32, { [1, C] }> = jj_1c % blk_1c;
        let jmod_bc: Tile<i32, { [16, C] }> = jmod_1c.broadcast(const_shape![16, C]);
        let brow_bc: Tile<i32, { [16, C] }> = iota_b
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, C]);
        let p_mask: Tile<bool, { [16, C] }> = eq_tile(jmod_bc, brow_bc);
        let p_mat: Tile<f32, { [16, C] }> = select(p_mask, one_bc, zero_bc);
        let p_mat_m: Tile<tf32, { [16, C] }> = convert_tile(p_mat);
        let x_c_m: Tile<tf32, { [C, 16] }> = convert_tile(x_c);
        let x_wide: Tile<f32, { [C, C] }> = mmaf(x_c_m, p_mat_m, zero_cc);
        let x_bd: Tile<f32, { [C, C] }> = select(same_block, x_wide, zero_cc);
        let x_bd_m: Tile<tf32, { [C, C] }> = convert_tile(x_bd);
        let a_off_m: Tile<tf32, { [C, C] }> = convert_tile(a_off);
        let n: Tile<f32, { [C, C] }> = mmaf(x_bd_m, a_off_m, zero_cc);
        let n_m: Tile<tf32, { [C, C] }> = convert_tile(n);
        let n2: Tile<f32, { [C, C] }> = mmaf(n_m, n_m, zero_cc);
        let i_minus_n: Tile<f32, { [C, C] }> = eye - n;
        let i_plus_n2: Tile<f32, { [C, C] }> = eye + n2;
        let i_minus_n_m: Tile<tf32, { [C, C] }> = convert_tile(i_minus_n);
        let i_plus_n2_m: Tile<tf32, { [C, C] }> = convert_tile(i_plus_n2);
        let p: Tile<f32, { [C, C] }> = mmaf(i_minus_n_m, i_plus_n2_m, zero_cc);
        let p_m: Tile<tf32, { [C, C] }> = convert_tile(p);
        let x: Tile<f32, { [C, C] }> = mmaf(p_m, x_bd_m, zero_cc);
        let ainv_m: Tile<bf16, { [C, C] }> = convert_tile(x);
        let egc: Tile<f32, { [C, 1] }> = exp(gcol);
        let beta_cv: Tile<f32, { [C, V] }> = beta_col.broadcast(const_shape![C, V]);
        let vb: Tile<f32, { [C, V] }> = vc * beta_cv;
        let vb_m: Tile<bf16, { [C, V] }> = convert_tile(vb);
        let u: Tile<f32, { [C, V] }> = mmaf(ainv_m, vb_m, zero_cv);
        let bg: Tile<f32, { [C, 1] }> = beta_col * egc;
        let bg_ck: Tile<f32, { [C, K] }> = bg.broadcast(const_shape![C, K]);
        let kb: Tile<f32, { [C, K] }> = kc * bg_ck;
        let kb_m: Tile<bf16, { [C, K] }> = convert_tile(kb);
        let w: Tile<f32, { [C, K] }> = mmaf(ainv_m, kb_m, zero_ck);

        let rows_c1: Tile<i32, { [C, 1] }> = iota_c.reshape(const_shape![C, 1]);
        let row0_ck: Tile<i32, { [C, K] }> = broadcast_scalar(idx * chunk, const_shape![C, K]);
        let rows_ck: Tile<i32, { [C, K] }> = rows_c1.broadcast(const_shape![C, K]) + row0_ck;
        let iota_k: Tile<i32, { [K] }> = iota(const_shape![K]);
        let cols_1k: Tile<i32, { [1, K] }> = iota_k.reshape(const_shape![1, K]);
        let cols_ck: Tile<i32, { [C, K] }> = cols_1k.broadcast(const_shape![C, K]);
        let kdim_ck: Tile<i32, { [C, K] }> = broadcast_scalar(k_dim, const_shape![C, K]);
        let w_off: Tile<i32, { [C, K] }> = rows_ck * kdim_ck + cols_ck;
        let w_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(w_ptr);
        let w_p1: PointerTile<*mut bf16, { [1, 1] }> = w_p0.reshape(const_shape![1, 1]);
        let w_p2: PointerTile<*mut bf16, { [C, K] }> = w_p1.broadcast(const_shape![C, K]);
        let w_ptrs: PointerTile<*mut bf16, { [C, K] }> = w_p2.offset_tile(w_off);
        let w_b: Tile<bf16, { [C, K] }> = convert_tile(w);
        store_ptr_tko(
            w_ptrs,
            w_b,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
        let row0_cv: Tile<i32, { [C, V] }> = broadcast_scalar(idx * chunk, const_shape![C, V]);
        let rows_cv: Tile<i32, { [C, V] }> = rows_c1.broadcast(const_shape![C, V]) + row0_cv;
        let iota_v: Tile<i32, { [V] }> = iota(const_shape![V]);
        let cols_1v: Tile<i32, { [1, V] }> = iota_v.reshape(const_shape![1, V]);
        let cols_cv: Tile<i32, { [C, V] }> = cols_1v.broadcast(const_shape![C, V]);
        let vdim_cv: Tile<i32, { [C, V] }> = broadcast_scalar(v_dim, const_shape![C, V]);
        let u_off: Tile<i32, { [C, V] }> = rows_cv * vdim_cv + cols_cv;
        let u_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(u_ptr);
        let u_p1: PointerTile<*mut bf16, { [1, 1] }> = u_p0.reshape(const_shape![1, 1]);
        let u_p2: PointerTile<*mut bf16, { [C, V] }> = u_p1.broadcast(const_shape![C, V]);
        let u_ptrs: PointerTile<*mut bf16, { [C, V] }> = u_p2.offset_tile(u_off);
        let u_b: Tile<bf16, { [C, V] }> = convert_tile(u);
        store_ptr_tko(
            u_ptrs,
            u_b,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }

    // Sequential over chunks per head: delta = u - w S (stored), the inter-chunk output term
    // e^gcum (q S) (stored token-major), S = e^gtot S + (k e^(gtot - gcum))^T delta on the value-major
    // pooled state row; padding rows (negative state row) write zeros.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn gdn_state<const C: i32, const K: i32, const V: i32>(
        out_ptr: *mut bf16,
        delta_ptr: *mut bf16,
        state_ptr: *mut f32,
        slots_ptr: *mut i32,
        q: &Tensor<bf16, { [-1, -1, -1] }>,
        k: &Tensor<bf16, { [-1, -1, -1] }>,
        w: &Tensor<bf16, { [-1, -1] }>,
        u: &Tensor<bf16, { [-1, -1] }>,
        g: &Tensor<f32, { [-1, -1] }>,
        num_chunks_f: f32,
        chunk: i32,
        seq_len_f: f32,
        k_dim: i32,
        v_dim: i32,
        kv_size: i32,
        num_heads: i32,
        out_row_stride: i32,
        has_slots_f: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bh: i32 = pid.0;
        let num_chunks_t: Tile<f32, { [1] }> = broadcast_scalar(num_chunks_f, const_shape![1]);
        let num_chunks_i: Tile<i32, { [1] }> = convert_tile(num_chunks_t);
        let num_chunks_s: Tile<i32, { [] }> = num_chunks_i.reshape(const_shape![]);
        let num_chunks: i32 = tile_to_scalar(num_chunks_s);
        let seq_len_t: Tile<f32, { [1] }> = broadcast_scalar(seq_len_f, const_shape![1]);
        let seq_len_i: Tile<i32, { [1] }> = convert_tile(seq_len_t);
        let seq_len_s: Tile<i32, { [] }> = seq_len_i.reshape(const_shape![]);
        let seq_len: i32 = tile_to_scalar(seq_len_s);
        let has_slots_t: Tile<f32, { [1] }> = broadcast_scalar(has_slots_f, const_shape![1]);
        let has_slots_i: Tile<i32, { [1] }> = convert_tile(has_slots_t);
        let has_slots_s: Tile<i32, { [] }> = has_slots_i.reshape(const_shape![]);
        let has_slots: i32 = tile_to_scalar(has_slots_s);

        let batch: i32 = bh / num_heads;
        let head: i32 = bh - batch * num_heads;
        let out_base: i32 = batch * seq_len * out_row_stride + head * v_dim;
        let pq = q.partition(const_shape![1, C, K]);
        let pk = k.partition(const_shape![1, C, K]);
        let pw = w.partition(const_shape![C, K]);
        let pu = u.partition(const_shape![C, V]);
        let pg = g.partition(const_shape![1, C]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        // pooled slots: state row = slot * heads + head, negative slot = padding; gathered: row = bh
        let mut state_row: i32 = bh;
        if has_slots != 0 {
            let r_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(slots_ptr);
            let r_p1: PointerTile<*mut i32, { [1] }> = r_p0.reshape(const_shape![1]);
            let batch_t: Tile<i32, { [1] }> = broadcast_scalar(batch, const_shape![1]);
            let r_p2: PointerTile<*mut i32, { [1] }> = r_p1.offset_tile(batch_t);
            let (slot_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                r_p2,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let slot_s: Tile<i32, { [] }> = slot_t.reshape(const_shape![]);
            let slot: i32 = tile_to_scalar(slot_s);
            if slot < 0 {
                state_row = -1;
            } else {
                state_row = slot * num_heads + head;
            }
        }

        let iota_k: Tile<i32, { [K] }> = iota(const_shape![K]);
        let iota_v: Tile<i32, { [V] }> = iota(const_shape![V]);
        let iota_c: Tile<i32, { [C] }> = iota(const_shape![C]);
        let cols_1v: Tile<i32, { [1, V] }> = iota_v.reshape(const_shape![1, V]);
        let rows_c1: Tile<i32, { [C, 1] }> = iota_c.reshape(const_shape![C, 1]);
        let rows_cv0: Tile<i32, { [C, V] }> = rows_c1.broadcast(const_shape![C, V]);
        let cols_cv: Tile<i32, { [C, V] }> = cols_1v.broadcast(const_shape![C, V]);
        let vdim_cv: Tile<i32, { [C, V] }> = broadcast_scalar(v_dim, const_shape![C, V]);
        let stride_cv: Tile<i32, { [C, V] }> = broadcast_scalar(out_row_stride, const_shape![C, V]);
        let obase_cv: Tile<i32, { [C, V] }> = broadcast_scalar(out_base, const_shape![C, V]);
        let seq_cv: Tile<i32, { [C, V] }> = broadcast_scalar(seq_len, const_shape![C, V]);
        let o_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let o_p1: PointerTile<*mut bf16, { [1, 1] }> = o_p0.reshape(const_shape![1, 1]);
        let o_p2: PointerTile<*mut bf16, { [C, V] }> = o_p1.broadcast(const_shape![C, V]);
        let d_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(delta_ptr);
        let d_p1: PointerTile<*mut bf16, { [1, 1] }> = d_p0.reshape(const_shape![1, 1]);
        let d_p2: PointerTile<*mut bf16, { [C, V] }> = d_p1.broadcast(const_shape![C, V]);
        let zero_cv: Tile<f32, { [C, V] }> = constant(0.0f32, const_shape![C, V]);
        let zero_cv_b: Tile<bf16, { [C, V] }> = constant(bf16::ZERO, const_shape![C, V]);

        if state_row >= 0 {
            // value-major storage: element (k, v) of a state row lives at v * K + k
            let rows_kv: Tile<i32, { [K, V] }> = iota_k
                .reshape(const_shape![K, 1])
                .broadcast(const_shape![K, V]);
            let cols_kv: Tile<i32, { [K, V] }> = cols_1v.broadcast(const_shape![K, V]);
            let kdim_kv: Tile<i32, { [K, V] }> = broadcast_scalar(k_dim, const_shape![K, V]);
            let base_kv: Tile<i32, { [K, V] }> =
                broadcast_scalar(state_row * kv_size, const_shape![K, V]);
            let s_off: Tile<i32, { [K, V] }> = cols_kv * kdim_kv + rows_kv + base_kv;
            let s_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(state_ptr);
            let s_p1: PointerTile<*mut f32, { [1, 1] }> = s_p0.reshape(const_shape![1, 1]);
            let s_p2: PointerTile<*mut f32, { [K, V] }> = s_p1.broadcast(const_shape![K, V]);
            let s_ptrs: PointerTile<*mut f32, { [K, V] }> = s_p2.offset_tile(s_off);
            let (s_init, _): (Tile<f32, { [K, V] }>, Token) = load_ptr_tko(
                s_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let mut s: Tile<f32, { [K, V] }> = s_init;
            for c in 0i32..num_chunks {
                let idx: i32 = bh * num_chunks + c;
                let wc_b: Tile<bf16, { [C, K] }> = pw.load([idx, 0]);
                let uc_b: Tile<bf16, { [C, V] }> = pu.load([idx, 0]);
                let qc3: Tile<bf16, { [1, C, K] }> = pq.load([bh, c, 0]);
                let qc_b: Tile<bf16, { [C, K] }> = qc3.reshape(const_shape![C, K]);
                let kc3: Tile<bf16, { [1, C, K] }> = pk.load([bh, c, 0]);
                let kc_b: Tile<bf16, { [C, K] }> = kc3.reshape(const_shape![C, K]);
                let g_row: Tile<f32, { [1, C] }> = pg.load([bh, c]);
                let s_m: Tile<bf16, { [K, V] }> = convert_tile(s);
                let uc: Tile<f32, { [C, V] }> = convert_tile(uc_b);
                let ws: Tile<f32, { [C, V] }> = mmaf(wc_b, s_m, zero_cv);
                let delta: Tile<f32, { [C, V] }> = uc - ws;
                let g1: Tile<f32, { [C] }> = g_row.reshape(const_shape![C]);
                let gcum: Tile<f32, { [C] }> =
                    scan(g1, 0i32, reverse::Forward, 0.0f32, |acc, x| acc + x);
                let gcol: Tile<f32, { [C, 1] }> = gcum.reshape(const_shape![C, 1]);
                let egc: Tile<f32, { [C, 1] }> = exp(gcol);
                let qs0: Tile<f32, { [C, V] }> = mmaf(qc_b, s_m, zero_cv);
                let qs: Tile<f32, { [C, V] }> = qs0 * egc.broadcast(const_shape![C, V]);
                let t0_cv: Tile<i32, { [C, V] }> = broadcast_scalar(c * chunk, const_shape![C, V]);
                let t_cv: Tile<i32, { [C, V] }> = rows_cv0 + t0_cv;
                let valid: Tile<bool, { [C, V] }> = lt_tile(t_cv, seq_cv);
                let o_off: Tile<i32, { [C, V] }> = t_cv * stride_cv + cols_cv + obase_cv;
                let o_ptrs: PointerTile<*mut bf16, { [C, V] }> = o_p2.offset_tile(o_off);
                let qs_b: Tile<bf16, { [C, V] }> = convert_tile(qs);
                store_ptr_tko(
                    o_ptrs,
                    qs_b,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(valid),
                    None,
                    Latency::<0>,
                );
                let row0_cv: Tile<i32, { [C, V] }> =
                    broadcast_scalar(idx * chunk, const_shape![C, V]);
                let d_off: Tile<i32, { [C, V] }> = (rows_cv0 + row0_cv) * vdim_cv + cols_cv;
                let d_ptrs: PointerTile<*mut bf16, { [C, V] }> = d_p2.offset_tile(d_off);
                let delta_b: Tile<bf16, { [C, V] }> = convert_tile(delta);
                store_ptr_tko(
                    d_ptrs,
                    delta_b,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    Latency::<0>,
                );
                let g_cc: Tile<f32, { [C, C] }> = g_row.broadcast(const_shape![C, C]);
                let gtot_c: Tile<f32, { [C] }> = reduce_sum(g_cc, 1i32);
                let g_kc: Tile<f32, { [K, C] }> = g_row.broadcast(const_shape![K, C]);
                let gtot_k: Tile<f32, { [K] }> = reduce_sum(g_kc, 1i32);
                let decay_last: Tile<f32, { [C, 1] }> =
                    exp(gtot_c.reshape(const_shape![C, 1]) - gcol);
                let sdecay: Tile<f32, { [K, V] }> =
                    exp(gtot_k.reshape(const_shape![K, 1])).broadcast(const_shape![K, V]);
                let kc: Tile<f32, { [C, K] }> = convert_tile(kc_b);
                let kd: Tile<f32, { [C, K] }> = kc * decay_last.broadcast(const_shape![C, K]);
                let kd_m: Tile<bf16, { [C, K] }> = convert_tile(kd);
                let kdt_m: Tile<bf16, { [K, C] }> = permute(kd_m, transpose);
                let s_decayed: Tile<f32, { [K, V] }> = s * sdecay;
                let s_new: Tile<f32, { [K, V] }> = mmaf(kdt_m, delta_b, s_decayed);
                s = s_new;
            }
            store_ptr_tko(
                s_ptrs,
                s,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            );
        } else {
            for c in 0i32..num_chunks {
                let idx: i32 = bh * num_chunks + c;
                let t0_cv: Tile<i32, { [C, V] }> = broadcast_scalar(c * chunk, const_shape![C, V]);
                let t_cv: Tile<i32, { [C, V] }> = rows_cv0 + t0_cv;
                let valid: Tile<bool, { [C, V] }> = lt_tile(t_cv, seq_cv);
                let o_off: Tile<i32, { [C, V] }> = t_cv * stride_cv + cols_cv + obase_cv;
                let o_ptrs: PointerTile<*mut bf16, { [C, V] }> = o_p2.offset_tile(o_off);
                store_ptr_tko(
                    o_ptrs,
                    zero_cv_b,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(valid),
                    None,
                    Latency::<0>,
                );
                let row0_cv: Tile<i32, { [C, V] }> =
                    broadcast_scalar(idx * chunk, const_shape![C, V]);
                let d_off: Tile<i32, { [C, V] }> = (rows_cv0 + row0_cv) * vdim_cv + cols_cv;
                let d_ptrs: PointerTile<*mut bf16, { [C, V] }> = d_p2.offset_tile(d_off);
                store_ptr_tko(
                    d_ptrs,
                    zero_cv_b,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    Latency::<0>,
                );
            }
        }
    }

    // o += (q k^T * D_incl) delta per (head, chunk) on the token-major output.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn gdn_out<const C: i32, const K: i32, const V: i32>(
        out_ptr: *mut bf16,
        q: &Tensor<bf16, { [-1, -1, -1] }>,
        k: &Tensor<bf16, { [-1, -1, -1] }>,
        delta: &Tensor<bf16, { [-1, -1] }>,
        g: &Tensor<f32, { [-1, -1] }>,
        num_chunks_f: f32,
        chunk: i32,
        seq_len_f: f32,
        v_dim: i32,
        num_heads: i32,
        out_row_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let idx: i32 = pid.0;
        let num_chunks_t: Tile<f32, { [1] }> = broadcast_scalar(num_chunks_f, const_shape![1]);
        let num_chunks_i: Tile<i32, { [1] }> = convert_tile(num_chunks_t);
        let num_chunks_s: Tile<i32, { [] }> = num_chunks_i.reshape(const_shape![]);
        let num_chunks: i32 = tile_to_scalar(num_chunks_s);
        let seq_len_t: Tile<f32, { [1] }> = broadcast_scalar(seq_len_f, const_shape![1]);
        let seq_len_i: Tile<i32, { [1] }> = convert_tile(seq_len_t);
        let seq_len_s: Tile<i32, { [] }> = seq_len_i.reshape(const_shape![]);
        let seq_len: i32 = tile_to_scalar(seq_len_s);
        let bh: i32 = idx / num_chunks;
        let c: i32 = idx - bh * num_chunks;
        let batch: i32 = bh / num_heads;
        let head: i32 = bh - batch * num_heads;
        let out_base: i32 = batch * seq_len * out_row_stride + head * v_dim;
        let pq = q.partition(const_shape![1, C, K]);
        let pk = k.partition(const_shape![1, C, K]);
        let pd = delta.partition(const_shape![C, V]);
        let pg = g.partition(const_shape![1, C]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let iota_c: Tile<i32, { [C] }> = iota(const_shape![C]);
        let ii: Tile<i32, { [C, C] }> = iota_c
            .reshape(const_shape![C, 1])
            .broadcast(const_shape![C, C]);
        let jj: Tile<i32, { [C, C] }> = iota_c
            .reshape(const_shape![1, C])
            .broadcast(const_shape![C, C]);
        let incl: Tile<bool, { [C, C] }> = le_tile(jj, ii);
        let zero_cc: Tile<f32, { [C, C] }> = constant(0.0f32, const_shape![C, C]);
        let g_row: Tile<f32, { [1, C] }> = pg.load([bh, c]);
        let g1: Tile<f32, { [C] }> = g_row.reshape(const_shape![C]);
        let gcum: Tile<f32, { [C] }> = scan(g1, 0i32, reverse::Forward, 0.0f32, |acc, x| acc + x);
        let gcol_cc: Tile<f32, { [C, C] }> = gcum
            .reshape(const_shape![C, 1])
            .broadcast(const_shape![C, C]);
        let grow_cc: Tile<f32, { [C, C] }> = gcum
            .reshape(const_shape![1, C])
            .broadcast(const_shape![C, C]);
        let dfull: Tile<f32, { [C, C] }> = exp(gcol_cc - grow_cc);
        let d_incl: Tile<f32, { [C, C] }> = select(incl, dfull, zero_cc);
        let qc3: Tile<bf16, { [1, C, K] }> = pq.load([bh, c, 0]);
        let qc_b: Tile<bf16, { [C, K] }> = qc3.reshape(const_shape![C, K]);
        let kc3: Tile<bf16, { [1, C, K] }> = pk.load([bh, c, 0]);
        let kc_b: Tile<bf16, { [C, K] }> = kc3.reshape(const_shape![C, K]);
        let dc_b: Tile<bf16, { [C, V] }> = pd.load([idx, 0]);
        let kt_b: Tile<bf16, { [K, C] }> = permute(kc_b, transpose);
        let qk0: Tile<f32, { [C, C] }> = mmaf(qc_b, kt_b, zero_cc);
        let qk: Tile<f32, { [C, C] }> = qk0 * d_incl;
        let qk_m: Tile<bf16, { [C, C] }> = convert_tile(qk);
        let rows_c1: Tile<i32, { [C, 1] }> = iota_c.reshape(const_shape![C, 1]);
        let t0_cv: Tile<i32, { [C, V] }> = broadcast_scalar(c * chunk, const_shape![C, V]);
        let t_cv: Tile<i32, { [C, V] }> = rows_c1.broadcast(const_shape![C, V]) + t0_cv;
        let iota_v: Tile<i32, { [V] }> = iota(const_shape![V]);
        let cols_cv: Tile<i32, { [C, V] }> = iota_v
            .reshape(const_shape![1, V])
            .broadcast(const_shape![C, V]);
        let stride_cv: Tile<i32, { [C, V] }> = broadcast_scalar(out_row_stride, const_shape![C, V]);
        let obase_cv: Tile<i32, { [C, V] }> = broadcast_scalar(out_base, const_shape![C, V]);
        let seq_cv: Tile<i32, { [C, V] }> = broadcast_scalar(seq_len, const_shape![C, V]);
        let valid: Tile<bool, { [C, V] }> = lt_tile(t_cv, seq_cv);
        let o_off: Tile<i32, { [C, V] }> = t_cv * stride_cv + cols_cv + obase_cv;
        let o_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let o_p1: PointerTile<*mut bf16, { [1, 1] }> = o_p0.reshape(const_shape![1, 1]);
        let o_p2: PointerTile<*mut bf16, { [C, V] }> = o_p1.broadcast(const_shape![C, V]);
        let o_ptrs: PointerTile<*mut bf16, { [C, V] }> = o_p2.offset_tile(o_off);
        let (o_loaded, _): (Tile<bf16, { [C, V] }>, Token) = load_ptr_tko(
            o_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid),
            None,
            None,
            Latency::<0>,
        );
        let zero_cv_b: Tile<bf16, { [C, V] }> = constant(bf16::ZERO, const_shape![C, V]);
        let o_partial_b: Tile<bf16, { [C, V] }> = select(valid, o_loaded, zero_cv_b);
        let o_partial: Tile<f32, { [C, V] }> = convert_tile(o_partial_b);
        let o: Tile<f32, { [C, V] }> = mmaf(qk_m, dc_b, o_partial);
        let o_b: Tile<bf16, { [C, V] }> = convert_tile(o);
        store_ptr_tko(
            o_ptrs,
            o_b,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid),
            None,
            Latency::<0>,
        );
    }
}

pub struct GdnPrefillKernel;

pub(super) static GDN_PREFILL: GdnPrefillKernel = GdnPrefillKernel;

/// `q`, `k`, `v` [BH, S_pad, 128] BF16 and `g`, `beta` [BH, S_pad] F32 with S_pad a multiple of
/// [`GDN_PREFILL_CHUNK`] and zero rows past `seq_len`; `state` F32 value-major rows of [128, 128]
/// updated in place, indexed per batch by the U32 `slots` (u32::MAX = padding) or by batch-head
/// order when `slots` is `None`.
pub struct GdnPrefillArgs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
    pub state: &'a Tensor,
    pub slots: Option<&'a Tensor>,
    pub batch_size: usize,
    pub num_heads: usize,
    pub seq_len: usize,
}

pub fn gdn_prefill_supported(dev: &CudaDevice, head_k_dim: usize, head_v_dim: usize) -> bool {
    jit_available(dev) && head_k_dim == GDN_PREFILL_HEAD_DIM && head_v_dim == GDN_PREFILL_HEAD_DIM
}

/// Returns the token-major output [B, S, H, 128] in BF16.
pub fn cutile_gdn_prefill(args: &GdnPrefillArgs<'_>, dev: &CudaDevice) -> Result<Tensor> {
    launch(args, dev, false)
}

fn borrow_bf16_3d(
    tensor: &Tensor,
    addr: u64,
    ordinal: usize,
) -> Result<cutile::tensor::Tensor<tile_bf16>> {
    let dims = tensor.dims3()?;
    let shape = vec![dims.0 as i32, dims.1 as i32, dims.2 as i32];
    let strides = vec![(dims.1 * dims.2) as i32, dims.2 as i32, 1];
    Ok(unsafe {
        cutile::tensor::Tensor::<tile_bf16>::borrow_raw_parts(
            addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    })
}

fn borrow_2d<T: cutile::DType>(
    rows: usize,
    cols: usize,
    addr: u64,
    ordinal: usize,
) -> cutile::tensor::Tensor<T> {
    unsafe {
        cutile::tensor::Tensor::<T>::borrow_raw_parts(
            addr as CUdeviceptr,
            ordinal,
            vec![rows as i32, cols as i32],
            vec![cols as i32, 1],
        )
    }
}

fn launch(args: &GdnPrefillArgs<'_>, dev: &CudaDevice, compile_only: bool) -> Result<Tensor> {
    let (bh, padded, k_dim) = args.q.dims3()?;
    let v_dim = args.v.dim(2)?;
    let seq_len = args.seq_len;
    if k_dim != GDN_PREFILL_HEAD_DIM || v_dim != GDN_PREFILL_HEAD_DIM {
        candle_core::bail!("cuTile GDN prefill needs K = V = {GDN_PREFILL_HEAD_DIM}")
    }
    if args.k.dims3()? != (bh, padded, k_dim)
        || args.v.dims3()? != (bh, padded, v_dim)
        || args.g.dims2()? != (bh, padded)
        || args.beta.dims2()? != (bh, padded)
        || args
            .slots
            .is_some_and(|slots| slots.dims1().ok() != Some(args.batch_size))
        || bh != args.batch_size * args.num_heads
        || seq_len == 0
        || padded < seq_len
        || !padded.is_multiple_of(GDN_PREFILL_CHUNK)
    {
        candle_core::bail!("cuTile GDN prefill got inconsistent operand shapes")
    }
    if args.q.dtype() != DType::BF16
        || args.k.dtype() != DType::BF16
        || args.v.dtype() != DType::BF16
        || args.g.dtype() != DType::F32
        || args.beta.dtype() != DType::F32
        || args.state.dtype() != DType::F32
        || args.slots.is_some_and(|slots| slots.dtype() != DType::U32)
    {
        candle_core::bail!("cuTile GDN prefill got unexpected operand dtypes")
    }
    let chunk = GDN_PREFILL_CHUNK;
    let num_chunks = padded / chunk;
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let mut w_buf = unsafe { dev.alloc::<bf16>(bh * padded * k_dim)? };
    let mut u_buf = unsafe { dev.alloc::<bf16>(bh * padded * v_dim)? };
    let mut delta_buf = unsafe { dev.alloc::<bf16>(bh * padded * v_dim)? };
    let mut out = unsafe { dev.alloc::<bf16>(args.batch_size * seq_len * args.num_heads * v_dim)? };

    for tensor in [args.q, args.k, args.v, args.g, args.beta, args.state]
        .into_iter()
        .chain(args.slots)
    {
        if !tensor.is_contiguous() {
            candle_core::bail!("cuTile GDN prefill needs contiguous operands")
        }
    }
    let (q_storage, q_layout) = args.q.storage_and_layout();
    let (k_storage, k_layout) = args.k.storage_and_layout();
    let (v_storage, v_layout) = args.v.storage_and_layout();
    let (g_storage, g_layout) = args.g.storage_and_layout();
    let (b_storage, b_layout) = args.beta.storage_and_layout();
    let (s_storage, s_layout) = args.state.storage_and_layout();
    let (
        Storage::Cuda(q_cuda),
        Storage::Cuda(k_cuda),
        Storage::Cuda(v_cuda),
        Storage::Cuda(g_cuda),
        Storage::Cuda(b_cuda),
        Storage::Cuda(s_cuda),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*b_storage,
        &*s_storage,
    )
    else {
        candle_core::bail!("cuTile GDN prefill operands must be CUDA tensors")
    };
    let slots_storage = args.slots.map(|slots| slots.storage_and_layout());
    let mut slots_guard = None;
    let mut slots_addr = 0u64;
    if let Some((storage, layout)) = &slots_storage {
        let Storage::Cuda(slots_cuda) = &**storage else {
            candle_core::bail!("cuTile GDN prefill slots must be a CUDA tensor")
        };
        let (addr, guard) = slice_ptr_on_stream(
            slots_cuda.as_cuda_slice::<u32>()?,
            layout.start_offset(),
            &stream,
        );
        slots_addr = addr;
        slots_guard = Some(guard);
    }
    let (q_addr, _q_guard) = slice_ptr_on_stream(
        q_cuda.as_cuda_slice::<bf16>()?,
        q_layout.start_offset(),
        &stream,
    );
    let (k_addr, _k_guard) = slice_ptr_on_stream(
        k_cuda.as_cuda_slice::<bf16>()?,
        k_layout.start_offset(),
        &stream,
    );
    let (v_addr, _v_guard) = slice_ptr_on_stream(
        v_cuda.as_cuda_slice::<bf16>()?,
        v_layout.start_offset(),
        &stream,
    );
    let (g_addr, _g_guard) = slice_ptr_on_stream(
        g_cuda.as_cuda_slice::<f32>()?,
        g_layout.start_offset(),
        &stream,
    );
    let (b_addr, _b_guard) = slice_ptr_on_stream(
        b_cuda.as_cuda_slice::<f32>()?,
        b_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<f32>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (w_addr, w_guard) = slice_ptr_mut_on_stream(&mut w_buf, 0, &stream);
    let (u_addr, u_guard) = slice_ptr_mut_on_stream(&mut u_buf, 0, &stream);
    let (d_addr, d_guard) = slice_ptr_mut_on_stream(&mut delta_buf, 0, &stream);
    let (o_addr, o_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

    let q_t = Arc::new(borrow_bf16_3d(args.q, q_addr, ordinal)?);
    let k_t = Arc::new(borrow_bf16_3d(args.k, k_addr, ordinal)?);
    let v_t = Arc::new(borrow_bf16_3d(args.v, v_addr, ordinal)?);
    let g_t = Arc::new(borrow_2d::<f32>(bh, padded, g_addr, ordinal));
    let b_t = Arc::new(borrow_2d::<f32>(bh, padded, b_addr, ordinal));
    let w_t = Arc::new(borrow_2d::<tile_bf16>(bh * padded, k_dim, w_addr, ordinal));
    let u_t = Arc::new(borrow_2d::<tile_bf16>(bh * padded, v_dim, u_addr, ordinal));
    let d_t = Arc::new(borrow_2d::<tile_bf16>(bh * padded, v_dim, d_addr, ordinal));
    // SAFETY: every pointer names a live candle allocation that outlives the launches on this stream.
    let (w_ptr, u_ptr, d_ptr, o_ptr, s_ptr, r_ptr) = unsafe {
        (
            DevicePointer::<tile_bf16>::from_cu_deviceptr(w_addr as CUdeviceptr),
            DevicePointer::<tile_bf16>::from_cu_deviceptr(u_addr as CUdeviceptr),
            DevicePointer::<tile_bf16>::from_cu_deviceptr(d_addr as CUdeviceptr),
            DevicePointer::<tile_bf16>::from_cu_deviceptr(o_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(s_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(slots_addr as CUdeviceptr),
        )
    };
    let generics = vec![
        chunk.to_string(),
        GDN_PREFILL_HEAD_DIM.to_string(),
        GDN_PREFILL_HEAD_DIM.to_string(),
    ];
    let nc = num_chunks as f32;
    let seq_len_f = seq_len as f32;
    let has_slots = if args.slots.is_some() { 1.0f32 } else { 0.0f32 };
    let out_row_stride = (args.num_heads * v_dim) as i32;
    let cutile_stream = context::stream(dev);
    let wy = unsafe {
        kernels::gdn_wy(
            w_ptr,
            u_ptr,
            k_t.clone(),
            v_t.clone(),
            g_t.clone(),
            b_t.clone(),
            nc,
            chunk as i32,
            k_dim as i32,
            v_dim as i32,
        )
    }
    .generics(generics.clone())
    .grid(((bh * num_chunks) as u32, 1, 1));
    let state = unsafe {
        kernels::gdn_state(
            o_ptr,
            d_ptr,
            s_ptr,
            r_ptr,
            q_t.clone(),
            k_t.clone(),
            w_t,
            u_t,
            g_t.clone(),
            nc,
            chunk as i32,
            seq_len_f,
            k_dim as i32,
            v_dim as i32,
            KV_SIZE as i32,
            args.num_heads as i32,
            out_row_stride,
            has_slots,
        )
    }
    .generics(generics.clone())
    .grid((bh as u32, 1, 1));
    let out_k = unsafe {
        kernels::gdn_out(
            o_ptr,
            q_t,
            k_t,
            d_t,
            g_t,
            nc,
            chunk as i32,
            seq_len_f,
            v_dim as i32,
            args.num_heads as i32,
            out_row_stride,
        )
    }
    .generics(generics)
    .grid(((bh * num_chunks) as u32, 1, 1));
    if compile_only {
        catch_cutile_panic("GDN prefill compile", || {
            wy.compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile gdn wy compile: {e:?}")))?;
            state
                .compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile gdn state compile: {e:?}")))?;
            out_k
                .compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile gdn out compile: {e:?}")))?;
            Ok(())
        })?;
    } else {
        catch_cutile_panic("GDN prefill launch", || unsafe {
            wy.async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile gdn wy launch: {e:?}")))?;
            state
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile gdn state launch: {e:?}")))?;
            out_k
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile gdn out launch: {e:?}")))?;
            Ok(())
        })?;
    }
    drop((w_guard, u_guard, d_guard, o_guard, slots_guard));
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[args.batch_size, seq_len, args.num_heads, v_dim]),
    )))
}

impl CutileKernel for GdnPrefillKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        if !jit_available(dev) {
            return Ok(());
        }
        let device = Device::Cuda(dev.clone());
        // 16 heads lands in the same divisibility class as real head counts (multiples of 16)
        let heads = 16;
        let seq_len = GDN_PREFILL_CHUNK;
        let q = Tensor::zeros((heads, seq_len, GDN_PREFILL_HEAD_DIM), DType::BF16, &device)?;
        let g = Tensor::zeros((heads, seq_len), DType::F32, &device)?;
        let state = Tensor::zeros(
            (heads, GDN_PREFILL_HEAD_DIM, GDN_PREFILL_HEAD_DIM),
            DType::F32,
            &device,
        )?;
        let slots = Tensor::zeros(1, DType::U32, &device)?;
        let args = GdnPrefillArgs {
            q: &q,
            k: &q,
            v: &q,
            g: &g,
            beta: &g,
            state: &state,
            slots: Some(&slots),
            batch_size: 1,
            num_heads: heads,
            seq_len,
        };
        tracing::info!("Warming cuTile GDN prefill kernels.");
        if let Err(err) = launch(&args, dev, true) {
            tracing::warn!("cuTile GDN prefill warmup failed: {err}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};
    use half::bf16;

    use super::{cutile_gdn_prefill, GdnPrefillArgs, GDN_PREFILL_CHUNK, GDN_PREFILL_HEAD_DIM};

    const D: usize = GDN_PREFILL_HEAD_DIM;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32
        }
    }

    // Per-token recurrence on a key-major [K, V] state: S *= e^g; delta = beta (v - k S); S += k^T delta; o = q S.
    fn reference(
        seq_len: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state: &mut [f32],
    ) -> Vec<f32> {
        let mut out = vec![0f32; seq_len * D];
        for t in 0..seq_len {
            let decay = g[t].exp();
            state.iter_mut().for_each(|x| *x *= decay);
            let kt = &k[t * D..(t + 1) * D];
            let mut kv = vec![0f32; D];
            for d in 0..D {
                for j in 0..D {
                    kv[j] += kt[d] * state[d * D + j];
                }
            }
            let delta: Vec<f32> = (0..D).map(|j| beta[t] * (v[t * D + j] - kv[j])).collect();
            for d in 0..D {
                for j in 0..D {
                    state[d * D + j] += kt[d] * delta[j];
                }
            }
            let qt = &q[t * D..(t + 1) * D];
            for j in 0..D {
                out[t * D + j] = (0..D).map(|d| qt[d] * state[d * D + j]).sum();
            }
        }
        out
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_gdn_prefill_matches_sequential_reference() -> Result<()> {
        const BATCH: usize = 3;
        const HEADS: usize = 2;
        const POOL_ROWS: usize = 9;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let mut rng = Lcg(11);
        // correlated keys with beta near 1 make (I + A)^-1 ill-conditioned like real activations
        for (seq_len, correlated) in [
            (1usize, false),
            (64, false),
            (100, false),
            (193, false),
            (64, true),
            (200, true),
        ] {
            let bh = BATCH * HEADS;
            let round = |x: f32| bf16::from_f32(x).to_f32();
            let mut q = vec![0f32; bh * seq_len * D];
            let mut k = vec![0f32; bh * seq_len * D];
            let mut v = vec![0f32; bh * seq_len * D];
            let base: Vec<f32> = (0..bh * D).map(|_| rng.next() - 0.5).collect();
            for row in 0..bh * seq_len {
                let mut norm = 0f32;
                for d in 0..D {
                    q[row * D + d] = round((rng.next() - 0.5) / (D as f32).sqrt());
                    let noise = rng.next() - 0.5;
                    let kv = if correlated {
                        base[(row / seq_len) * D + d] + 0.3 * noise
                    } else {
                        noise
                    };
                    k[row * D + d] = kv;
                    norm += kv * kv;
                }
                for d in 0..D {
                    k[row * D + d] = round(k[row * D + d] / norm.sqrt());
                }
                for j in 0..D {
                    v[row * D + j] = round(rng.next() - 0.5);
                }
            }
            let (g_scale, beta_floor) = if correlated {
                (0.15, 0.85)
            } else {
                (0.05, 0.0)
            };
            let g: Vec<f32> = (0..bh * seq_len).map(|_| -g_scale * rng.next()).collect();
            let beta: Vec<f32> = (0..bh * seq_len)
                .map(|_| beta_floor + (1.0 - beta_floor) * rng.next())
                .collect();
            // value-major pool rows [slot * heads + head][v][k]; batch 1 is padding
            let pool: Vec<f32> = (0..POOL_ROWS * D * D)
                .map(|_| 0.05 * (rng.next() - 0.5))
                .collect();
            let slots: Vec<u32> = vec![3, u32::MAX, 0];
            let rows: Vec<i32> = (0..bh)
                .map(|idx| match slots[idx / HEADS] {
                    u32::MAX => -1,
                    slot => (slot as usize * HEADS + idx % HEADS) as i32,
                })
                .collect();
            let s_pad = seq_len.div_ceil(GDN_PREFILL_CHUNK) * GDN_PREFILL_CHUNK;
            let pad_rows = |data: &[f32], width: usize| {
                let mut out = vec![0f32; bh * s_pad * width];
                for idx in 0..bh {
                    for t in 0..seq_len {
                        let src = (idx * seq_len + t) * width;
                        let dst = (idx * s_pad + t) * width;
                        out[dst..dst + width].copy_from_slice(&data[src..src + width]);
                    }
                }
                out
            };
            let to_bf16 =
                |data: &[f32]| data.iter().map(|x| bf16::from_f32(*x)).collect::<Vec<_>>();
            let q_t = Tensor::from_vec(to_bf16(&pad_rows(&q, D)), (bh, s_pad, D), &dev)?;
            let k_t = Tensor::from_vec(to_bf16(&pad_rows(&k, D)), (bh, s_pad, D), &dev)?;
            let v_t = Tensor::from_vec(to_bf16(&pad_rows(&v, D)), (bh, s_pad, D), &dev)?;
            let g_t = Tensor::from_vec(pad_rows(&g, 1), (bh, s_pad), &dev)?;
            let beta_t = Tensor::from_vec(pad_rows(&beta, 1), (bh, s_pad), &dev)?;
            let state_t = Tensor::from_vec(pool.clone(), (POOL_ROWS, D, D), &dev)?;
            let slots_t = Tensor::from_vec(slots.clone(), BATCH, &dev)?;
            let out = cutile_gdn_prefill(
                &GdnPrefillArgs {
                    q: &q_t,
                    k: &k_t,
                    v: &v_t,
                    g: &g_t,
                    beta: &beta_t,
                    state: &state_t,
                    slots: Some(&slots_t),
                    batch_size: BATCH,
                    num_heads: HEADS,
                    seq_len,
                },
                cuda,
            )?;
            assert_eq!(out.dims(), &[BATCH, seq_len, HEADS, D]);
            let out_host = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let state_host = state_t.flatten_all()?.to_vec1::<f32>()?;
            for (idx, &row) in rows.iter().enumerate() {
                let (b, h) = (idx / HEADS, idx % HEADS);
                let tok = idx * seq_len;
                let out_bh: Vec<f32> = (0..seq_len)
                    .flat_map(|t| {
                        let base = ((b * seq_len + t) * HEADS + h) * D;
                        out_host[base..base + D].to_vec()
                    })
                    .collect();
                if row < 0 {
                    assert!(
                        out_bh.iter().all(|x| *x == 0.0),
                        "seq_len={seq_len} padding batch must emit zeros"
                    );
                    continue;
                }
                // key-major copy of the value-major pool row for the reference
                let pool_row = &pool[row as usize * D * D..(row as usize + 1) * D * D];
                let mut state_ref = vec![0f32; D * D];
                for kk in 0..D {
                    for vv in 0..D {
                        state_ref[kk * D + vv] = pool_row[vv * D + kk];
                    }
                }
                let out_ref = reference(
                    seq_len,
                    &q[tok * D..(tok + seq_len) * D],
                    &k[tok * D..(tok + seq_len) * D],
                    &v[tok * D..(tok + seq_len) * D],
                    &g[tok..tok + seq_len],
                    &beta[tok..tok + seq_len],
                    &mut state_ref,
                );
                let max_err = out_bh
                    .iter()
                    .zip(&out_ref)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let max_ref = out_ref.iter().map(|x| x.abs()).fold(0f32, f32::max);
                assert!(
                    max_err <= 2.0e-2 * max_ref.max(1.0e-2),
                    "seq_len={seq_len} correlated={correlated} bh={idx}: output error {max_err} vs {max_ref}"
                );
                let got_row = &state_host[row as usize * D * D..(row as usize + 1) * D * D];
                let mut state_err = 0f32;
                let mut state_max = 0f32;
                for kk in 0..D {
                    for vv in 0..D {
                        state_err =
                            state_err.max((got_row[vv * D + kk] - state_ref[kk * D + vv]).abs());
                        state_max = state_max.max(state_ref[kk * D + vv].abs());
                    }
                }
                assert!(
                    state_err <= 2.0e-2 * state_max.max(1.0e-2),
                    "seq_len={seq_len} correlated={correlated} bh={idx}: state error {state_err} vs {state_max}"
                );
            }
        }
        Ok(())
    }
}
