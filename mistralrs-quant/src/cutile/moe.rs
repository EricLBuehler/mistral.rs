//! cuTile transcription of vLLM's Triton `fused_moe_kernel` (unquantized BF16).
//!
//! Faithful 1:1 port. Weights use the natural stacked `[E, N, K]` = `[E, out, in]` layout (vLLM
//! canonical); B is read directly via strides (K contiguous). A and B are both gathered
//! via masked pointer tiles (Triton's `a_ptrs`/`b_ptrs` are pointer arithmetic).
//! Launched twice: gate_up (N=2*I, top_k=model top-k, mul=0) then down (N=hidden, top_k=1, mul=1).
//!
//! Every PointerTile/Tile intermediate is explicitly annotated: the cuTile JIT
//! type-inference needs a compiled tile type at each step (fluent chains fail with
//! "return type is missing a compiled tile type").

#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

#[cutile::module]
pub mod fused_moe {
    use cutile::core::*;

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
        )
    )]
    pub unsafe fn fused_moe_kernel<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_M: i32,
        const TOP_K: i32,             // model top-k for gate_up, 1 for down
        const MUL_ROUTED_WEIGHT: i32, // 0 / 1
    >(
        out_ptr: *mut bf16,                   // C: [num_valid_tokens, N], stride (N, 1)
        a_ptr: *mut bf16,                     // A: [num_a_rows, K], stride (K, 1)
        b_ptr: *mut bf16,                     // B: [E, N, K], stride (N*K, K, 1)
        sorted_token_ids_ptr: *mut i32,       // [EM]
        expert_ids_ptr: *mut i32,             // [num_pid_m]
        num_tokens_post_padded_ptr: *mut i32, // scalar
        topk_weights_ptr: *mut f32,           // [num_valid_tokens]
        n_size: i32,
        k_size: i32,
        em: i32,
        num_valid_tokens: i32,
    ) {
        let pid: i32 = get_tile_block_id().0;
        let num_pid_m: i32 = ceil_div(em, BM);
        let num_pid_n: i32 = ceil_div(n_size, BN);
        let num_pid_in_group: i32 = GROUP_M * num_pid_n;
        let group_id: i32 = pid / num_pid_in_group;
        let first_pid_m: i32 = group_id * GROUP_M;
        let group_size_m: i32 = {
            let rem = num_pid_m - first_pid_m;
            if rem < GROUP_M {
                rem
            } else {
                GROUP_M
            }
        };
        let pid_m: i32 = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
        let pid_n: i32 = (pid % num_pid_in_group) / group_size_m;

        // num_tokens_post_padded (scalar load)
        let ntpp_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(num_tokens_post_padded_ptr);
        let ntpp_p1: PointerTile<*mut i32, { [1] }> = ntpp_p0.reshape(const_shape![1]);
        let (ntpp_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            ntpp_p1,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let ntpp_s: Tile<i32, { [] }> = ntpp_t.reshape(const_shape![]);
        let ntpp: i32 = tile_to_scalar(ntpp_s);

        if pid_m * BM < ntpp {
            // offs_token = sorted_token_ids[pid_m*BM + arange(BM)]   (contiguous, sentinel fill)
            let iota_m: Tile<i32, { [BM] }> = iota(const_shape![BM]);
            let base_m: Tile<i32, { [BM] }> = broadcast_scalar(pid_m * BM, const_shape![BM]);
            let offs_token_id: Tile<i32, { [BM] }> = iota_m + base_m;
            let em_t: Tile<i32, { [BM] }> = broadcast_scalar(em, const_shape![BM]);
            let id_inb: Tile<bool, { [BM] }> = lt_tile(offs_token_id, em_t);

            let sids_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(sorted_token_ids_ptr);
            let sids_p1: PointerTile<*mut i32, { [1] }> = sids_p0.reshape(const_shape![1]);
            let sids_p2: PointerTile<*mut i32, { [BM] }> = sids_p1.broadcast(const_shape![BM]);
            let sids_ptrs: PointerTile<*mut i32, { [BM] }> = sids_p2.offset_tile(offs_token_id);
            let (offs_token, _): (Tile<i32, { [BM] }>, Token) = load_ptr_tko(
                sids_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(id_inb),
                Some(num_valid_tokens),
                None,
                Latency::<0>,
            );
            let nvt_t: Tile<i32, { [BM] }> = broadcast_scalar(num_valid_tokens, const_shape![BM]);
            let token_mask: Tile<bool, { [BM] }> = lt_tile(offs_token, nvt_t);

            // off_experts = expert_ids[pid_m]   (scalar)
            let eid_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(expert_ids_ptr);
            let eid_p1: PointerTile<*mut i32, { [1] }> = eid_p0.reshape(const_shape![1]);
            let pid_m_t: Tile<i32, { [1] }> = broadcast_scalar(pid_m, const_shape![1]);
            let eid_p2: PointerTile<*mut i32, { [1] }> = eid_p1.offset_tile(pid_m_t);
            let (eid_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                eid_p2,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let eid_s: Tile<i32, { [] }> = eid_t.reshape(const_shape![]);
            let off_experts: i32 = tile_to_scalar(eid_s);

            let iota_n: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let base_n: Tile<i32, { [BN] }> = broadcast_scalar(pid_n * BN, const_shape![BN]);
            let offs_cn: Tile<i32, { [BN] }> = iota_n + base_n;
            let n_t_bn: Tile<i32, { [BN] }> = broadcast_scalar(n_size, const_shape![BN]);
            let cn_inb: Tile<bool, { [BN] }> = lt_tile(offs_cn, n_t_bn);

            let ot_col: Tile<i32, { [BM, 1] }> = offs_token.reshape(const_shape![BM, 1]);
            let ot_2d: Tile<i32, { [BM, BN] }> = ot_col.broadcast(const_shape![BM, BN]);
            let n_2d: Tile<i32, { [BM, BN] }> = broadcast_scalar(n_size, const_shape![BM, BN]);
            let ot_n: Tile<i32, { [BM, BN] }> = muli(ot_2d, n_2d, overflow::NoSignedWrap);
            let cn_row: Tile<i32, { [1, BN] }> = offs_cn.reshape(const_shape![1, BN]);
            let cn_2d: Tile<i32, { [BM, BN] }> = cn_row.broadcast(const_shape![BM, BN]);
            let c_off: Tile<i32, { [BM, BN] }> = ot_n + cn_2d;
            let c_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
            let c_base1: PointerTile<*mut bf16, { [1, 1] }> = c_base0.reshape(const_shape![1, 1]);
            let c_base2: PointerTile<*mut bf16, { [BM, BN] }> =
                c_base1.broadcast(const_shape![BM, BN]);
            let c_ptrs: PointerTile<*mut bf16, { [BM, BN] }> = c_base2.offset_tile(c_off);

            let tm_col: Tile<bool, { [BM, 1] }> = token_mask.reshape(const_shape![BM, 1]);
            let tm_2d: Tile<bool, { [BM, BN] }> = tm_col.broadcast(const_shape![BM, BN]);
            let cnb_row: Tile<bool, { [1, BN] }> = cn_inb.reshape(const_shape![1, BN]);
            let cnb_2d: Tile<bool, { [BM, BN] }> = cnb_row.broadcast(const_shape![BM, BN]);
            let c_mask: Tile<bool, { [BM, BN] }> = tm_2d & cnb_2d;

            if off_experts == -1 {
                let zeros: Tile<bf16, { [BM, BN] }> = constant(bf16::ZERO, const_shape![BM, BN]);
                store_ptr_tko(
                    c_ptrs,
                    zeros,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(c_mask),
                    None,
                    Latency::<0>,
                );
            } else {
                // offs_bn = offs_cn % N bounds B reads on the last N-tile; manual modulo (cuTile remi won't serialize).
                let n_tile: Tile<i32, { [BN] }> = broadcast_scalar(n_size, const_shape![BN]);
                let q_bn: Tile<i32, { [BN] }> = offs_cn / n_tile;
                let qn_bn: Tile<i32, { [BN] }> = muli(q_bn, n_tile, overflow::NoSignedWrap);
                let offs_bn: Tile<i32, { [BN] }> = subi(offs_cn, qn_bn, overflow::NoSignedWrap);
                let top_k_t: Tile<i32, { [BM] }> = broadcast_scalar(TOP_K, const_shape![BM]);
                let a_row: Tile<i32, { [BM] }> = offs_token / top_k_t;
                // Clamp padding rows (token_mask false) to row 0 so gathered A addresses stay
                // in-bounds; those rows are discarded at the C store (c_mask includes token_mask).
                let zero_row: Tile<i32, { [BM] }> = broadcast_scalar(0i32, const_shape![BM]);
                let safe_row: Tile<i32, { [BM] }> = select(token_mask, a_row, zero_row);
                let k_t_bm: Tile<i32, { [BM] }> = broadcast_scalar(k_size, const_shape![BM]);
                let a_row_off: Tile<i32, { [BM] }> = muli(safe_row, k_t_bm, overflow::NoSignedWrap);
                let be: i32 = off_experts * (k_size * n_size);
                let a_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(a_ptr);
                let a_base1: PointerTile<*mut bf16, { [1, 1] }> =
                    a_base0.reshape(const_shape![1, 1]);
                let a_base2: PointerTile<*mut bf16, { [BM, BK] }> =
                    a_base1.broadcast(const_shape![BM, BK]);
                let b_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(b_ptr);
                let b_base1: PointerTile<*mut bf16, { [1, 1] }> =
                    b_base0.reshape(const_shape![1, 1]);
                let b_base2: PointerTile<*mut bf16, { [BK, BN] }> =
                    b_base1.broadcast(const_shape![BK, BN]);

                let iota_k: Tile<i32, { [BK] }> = iota(const_shape![BK]);
                let ar_col: Tile<i32, { [BM, 1] }> = a_row_off.reshape(const_shape![BM, 1]);
                let ar_2d: Tile<i32, { [BM, BK] }> = ar_col.broadcast(const_shape![BM, BK]);
                let ok_row: Tile<i32, { [1, BK] }> = iota_k.reshape(const_shape![1, BK]);
                let ok_2d_a: Tile<i32, { [BM, BK] }> = ok_row.broadcast(const_shape![BM, BK]);
                let a_off: Tile<i32, { [BM, BK] }> = ar_2d + ok_2d_a;
                let mut a_ptrs: PointerTile<*mut bf16, { [BM, BK] }> = a_base2.offset_tile(a_off);

                // ENK [E, N, K]: B[e,n,k] = be + n*k_size + k (K contiguous, N stride k_size); natural [out,in] weight.
                let be_2d: Tile<i32, { [BK, BN] }> = broadcast_scalar(be, const_shape![BK, BN]);
                let ok_col: Tile<i32, { [BK, 1] }> = iota_k.reshape(const_shape![BK, 1]);
                let ok_2d_b: Tile<i32, { [BK, BN] }> = ok_col.broadcast(const_shape![BK, BN]);
                let obn_row: Tile<i32, { [1, BN] }> = offs_bn.reshape(const_shape![1, BN]);
                let obn_2d: Tile<i32, { [BK, BN] }> = obn_row.broadcast(const_shape![BK, BN]);
                let k_2d_b: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(k_size, const_shape![BK, BN]);
                let obn_k: Tile<i32, { [BK, BN] }> = muli(obn_2d, k_2d_b, overflow::NoSignedWrap);
                let b_off_a: Tile<i32, { [BK, BN] }> = be_2d + obn_k;
                let b_off: Tile<i32, { [BK, BN] }> = b_off_a + ok_2d_b;
                let mut b_ptrs: PointerTile<*mut bf16, { [BK, BN] }> = b_base2.offset_tile(b_off);
                let a_step: Tile<i32, { [BM, BK] }> = broadcast_scalar(BK, const_shape![BM, BK]);
                let b_step: Tile<i32, { [BK, BN] }> = broadcast_scalar(BK, const_shape![BK, BN]);

                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let kt: i32 = ceil_div(k_size, BK);
                for kk in 0i32..kt {
                    let base_k: Tile<i32, { [BK] }> = broadcast_scalar(kk * BK, const_shape![BK]);
                    let offs_k: Tile<i32, { [BK] }> = iota_k + base_k;
                    let k_size_t: Tile<i32, { [BK] }> = broadcast_scalar(k_size, const_shape![BK]);
                    let k_mask_1d: Tile<bool, { [BK] }> = lt_tile(offs_k, k_size_t);
                    let a_mask_row: Tile<bool, { [1, BK] }> =
                        k_mask_1d.reshape(const_shape![1, BK]);
                    let a_mask: Tile<bool, { [BM, BK] }> =
                        a_mask_row.broadcast(const_shape![BM, BK]);
                    let b_mask_col: Tile<bool, { [BK, 1] }> =
                        k_mask_1d.reshape(const_shape![BK, 1]);
                    let b_k_mask: Tile<bool, { [BK, BN] }> =
                        b_mask_col.broadcast(const_shape![BK, BN]);
                    let b_n_mask_row: Tile<bool, { [1, BN] }> = cn_inb.reshape(const_shape![1, BN]);
                    let b_n_mask: Tile<bool, { [BK, BN] }> =
                        b_n_mask_row.broadcast(const_shape![BK, BN]);
                    let b_mask: Tile<bool, { [BK, BN] }> = b_k_mask & b_n_mask;
                    let (a_load, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                        a_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(a_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let (b_load, _): (Tile<bf16, { [BK, BN] }>, Token) = load_ptr_tko(
                        b_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(b_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let a_zero: Tile<bf16, { [BM, BK] }> =
                        constant(bf16::ZERO, const_shape![BM, BK]);
                    let b_zero: Tile<bf16, { [BK, BN] }> =
                        constant(bf16::ZERO, const_shape![BK, BN]);
                    let a_raw: Tile<bf16, { [BM, BK] }> = select(a_mask, a_load, a_zero);
                    let b_tile: Tile<bf16, { [BK, BN] }> = select(b_mask, b_load, b_zero);

                    acc = mmaf(a_raw, b_tile, acc);
                    a_ptrs = a_ptrs.offset_tile(a_step);
                    b_ptrs = b_ptrs.offset_tile(b_step);
                }

                if MUL_ROUTED_WEIGHT != 0 {
                    let w_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(topk_weights_ptr);
                    let w_p1: PointerTile<*mut f32, { [1] }> = w_p0.reshape(const_shape![1]);
                    let w_p2: PointerTile<*mut f32, { [BM] }> = w_p1.broadcast(const_shape![BM]);
                    let w_ptrs: PointerTile<*mut f32, { [BM] }> = w_p2.offset_tile(offs_token);
                    let (moe_w, _): (Tile<f32, { [BM] }>, Token) = load_ptr_tko(
                        w_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(token_mask),
                        Some(0.0f32),
                        None,
                        Latency::<0>,
                    );
                    let moe_w_col: Tile<f32, { [BM, 1] }> = moe_w.reshape(const_shape![BM, 1]);
                    let moe_w_2d: Tile<f32, { [BM, BN] }> =
                        moe_w_col.broadcast(const_shape![BM, BN]);
                    acc = acc * moe_w_2d;
                }

                let acc_bf: Tile<bf16, { [BM, BN] }> = convert_tile(acc);
                store_ptr_tko(
                    c_ptrs,
                    acc_bf,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(c_mask),
                    None,
                    Latency::<0>,
                );
            }
        }
    }
}
