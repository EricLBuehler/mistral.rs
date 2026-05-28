//! cuTile transcription of vLLM's Triton `fused_moe_kernel` (unquantized BF16).
//!
//! Faithful 1:1 port. Weights use mistral.rs stacked `[E, K, N]` layout (in, out)
//! so `stride_be=K*N, stride_bk=N, stride_bn=1`; the mma `sum_k A[m,k]*B[e,k,n]`
//! is identical math to vLLM's transposed `[E,N,K]` read. A and B are both gathered
//! via masked pointer tiles (Triton's `a_ptrs`/`b_ptrs` are pointer arithmetic).
//! Launched twice: gate_up (N=2*I, top_k=8, mul=0) then down (N=hidden, top_k=1, mul=1).
//!
//! Every PointerTile/Tile intermediate is explicitly annotated: the cuTile JIT
//! type-inference needs a compiled tile type at each step (fluent chains fail with
//! "return type is missing a compiled tile type").

#[cutile::module]
pub mod fused_moe {
    use cutile::core::*;

    #[cutile::entry()]
    pub unsafe fn fused_moe_kernel<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_M: i32,
        const TOP_K: i32,             // 8 for gate_up GEMM, 1 for down GEMM
        const MUL_ROUTED_WEIGHT: i32, // 0 / 1
    >(
        out_ptr: *mut T,                   // C: [num_valid_tokens, N], stride (N, 1)
        a_ptr: *mut T,                     // A: [num_a_rows, K], stride (K, 1)
        b_ptr: *mut T,                     // B: [E, K, N], stride (K*N, N, 1)
        sorted_token_ids_ptr: *mut i32,       // [EM]
        expert_ids_ptr: *mut i32,             // [num_pid_m]
        num_tokens_post_padded_ptr: *mut i32, // scalar
        topk_weights_ptr: *mut f32,           // [num_valid_tokens]
        elem_bytes: i64,                      // size_of::<T>(); 2D pointer tiles built via int addresses
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

            // C pointers + mask: c[m,n] = out + offs_token[m]*N + offs_cn[n]
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
            // 2D pointer tile via int addresses (broadcast_ptr requires equal rank; rank0->2 fails verify).
            let out_i64p: PointerTile<*mut i64, { [] }> = ptr_to_ptr(pointer_to_tile(out_ptr));
            let out_base: i64 = tile_to_scalar(ptr_to_int(out_i64p));
            let c_base: Tile<i64, { [BM, BN] }> = broadcast_scalar(out_base, const_shape![BM, BN]);
            let c_off64: Tile<i64, { [BM, BN] }> = exti(c_off);
            let eb_cn: Tile<i64, { [BM, BN] }> = broadcast_scalar(elem_bytes, const_shape![BM, BN]);
            let c_byte: Tile<i64, { [BM, BN] }> = muli(c_off64, eb_cn, overflow::NoSignedWrap);
            let c_addr: Tile<i64, { [BM, BN] }> = c_base + c_byte;
            let c_ptrs: PointerTile<*mut T, { [BM, BN] }> = int_to_ptr(c_addr);

            let tm_col: Tile<bool, { [BM, 1] }> = token_mask.reshape(const_shape![BM, 1]);
            let tm_2d: Tile<bool, { [BM, BN] }> = tm_col.broadcast(const_shape![BM, BN]);
            let cnb_row: Tile<bool, { [1, BN] }> = cn_inb.reshape(const_shape![1, BN]);
            let cnb_2d: Tile<bool, { [BM, BN] }> = cnb_row.broadcast(const_shape![BM, BN]);
            let c_mask: Tile<bool, { [BM, BN] }> = tm_2d & cnb_2d;

            if off_experts == -1 {
                let zeros: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
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
                // offs_bn = (pid_n*BN + arange) % N; for Gemma N % BN == 0 so offs_cn < N always
                // and the modulo is a no-op. (remi has no signedness attr -> bytecode writer rejects it.)
                let offs_bn: Tile<i32, { [BN] }> = offs_cn;
                // a_row = offs_token // TOP_K. TOP_K is 1 or a power of 2, and offs_token >= 0,
                // so floor-div == arithmetic shift right by log2(TOP_K). (divi/remi omit the
                // signedness attr -> bytecode writer rejects them; shri carries it.)
                let shift_k: i32 = if TOP_K == 1 {
                    0
                } else if TOP_K == 2 {
                    1
                } else if TOP_K == 4 {
                    2
                } else {
                    3
                };
                let shift_t: Tile<i32, { [BM] }> = broadcast_scalar(shift_k, const_shape![BM]);
                let a_row: Tile<i32, { [BM] }> = shri(offs_token, shift_t);
                // Clamp padding rows (token_mask false) to row 0 so gathered A addresses stay
                // in-bounds; those rows are discarded at the C store (c_mask includes token_mask).
                let zero_row: Tile<i32, { [BM] }> = broadcast_scalar(0i32, const_shape![BM]);
                let safe_row: Tile<i32, { [BM] }> = select(token_mask, a_row, zero_row);
                let k_t_bm: Tile<i32, { [BM] }> = broadcast_scalar(k_size, const_shape![BM]);
                let a_row_off: Tile<i32, { [BM] }> = muli(safe_row, k_t_bm, overflow::NoSignedWrap);
                let be: i32 = off_experts * (k_size * n_size);
                let a_i64p: PointerTile<*mut i64, { [] }> = ptr_to_ptr(pointer_to_tile(a_ptr));
                let a_base_s: i64 = tile_to_scalar(ptr_to_int(a_i64p));
                let b_i64p: PointerTile<*mut i64, { [] }> = ptr_to_ptr(pointer_to_tile(b_ptr));
                let b_base_s: i64 = tile_to_scalar(ptr_to_int(b_i64p));

                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let kt: i32 = ceil_div(k_size, BK);
                for kk in 0i32..kt {
                    let iota_k: Tile<i32, { [BK] }> = iota(const_shape![BK]);
                    let base_k: Tile<i32, { [BK] }> = broadcast_scalar(kk * BK, const_shape![BK]);
                    let offs_k: Tile<i32, { [BK] }> = iota_k + base_k;
                    let k_t_bk: Tile<i32, { [BK] }> = broadcast_scalar(k_size, const_shape![BK]);
                    let kvalid: Tile<bool, { [BK] }> = lt_tile(offs_k, k_t_bk);
                    // Clamp K-edge columns to 0 so addresses stay in-bounds; A's K-edge lanes are
                    // zeroed after load (zeroing A nulls the dot term, so B needs no K mask).
                    let zero_k: Tile<i32, { [BK] }> = broadcast_scalar(0i32, const_shape![BK]);
                    let safe_k: Tile<i32, { [BK] }> = select(kvalid, offs_k, zero_k);

                    // A gather: a[m,k] = a_ptr + safe_row[m]*K + safe_k[k]
                    let ar_col: Tile<i32, { [BM, 1] }> = a_row_off.reshape(const_shape![BM, 1]);
                    let ar_2d: Tile<i32, { [BM, BK] }> = ar_col.broadcast(const_shape![BM, BK]);
                    let ok_row: Tile<i32, { [1, BK] }> = safe_k.reshape(const_shape![1, BK]);
                    let ok_2d_a: Tile<i32, { [BM, BK] }> = ok_row.broadcast(const_shape![BM, BK]);
                    let a_off: Tile<i32, { [BM, BK] }> = ar_2d + ok_2d_a;
                    let a_base: Tile<i64, { [BM, BK] }> = broadcast_scalar(a_base_s, const_shape![BM, BK]);
                    let a_off64: Tile<i64, { [BM, BK] }> = exti(a_off);
                    let eb_a: Tile<i64, { [BM, BK] }> = broadcast_scalar(elem_bytes, const_shape![BM, BK]);
                    let a_byte: Tile<i64, { [BM, BK] }> = muli(a_off64, eb_a, overflow::NoSignedWrap);
                    let a_addr: Tile<i64, { [BM, BK] }> = a_base + a_byte;
                    let a_ptrs: PointerTile<*mut T, { [BM, BK] }> = int_to_ptr(a_addr);
                    let (a_raw, _): (Tile<T, { [BM, BK] }>, Token) = load_ptr_tko(
                        a_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let kv_a_row: Tile<bool, { [1, BK] }> = kvalid.reshape(const_shape![1, BK]);
                    let kv_a: Tile<bool, { [BM, BK] }> = kv_a_row.broadcast(const_shape![BM, BK]);
                    let zeros_a: Tile<T, { [BM, BK] }> = constant(T::ZERO, const_shape![BM, BK]);
                    let a_tile: Tile<T, { [BM, BK] }> = select(kv_a, a_raw, zeros_a);

                    // B gather: b[k,n] = b_ptr + be + safe_k[k]*N + offs_bn[n]
                    let be_2d: Tile<i32, { [BK, BN] }> = broadcast_scalar(be, const_shape![BK, BN]);
                    let ok_col: Tile<i32, { [BK, 1] }> = safe_k.reshape(const_shape![BK, 1]);
                    let ok_2d_b: Tile<i32, { [BK, BN] }> = ok_col.broadcast(const_shape![BK, BN]);
                    let n_2d_b: Tile<i32, { [BK, BN] }> = broadcast_scalar(n_size, const_shape![BK, BN]);
                    let ok_n: Tile<i32, { [BK, BN] }> = muli(ok_2d_b, n_2d_b, overflow::NoSignedWrap);
                    let obn_row: Tile<i32, { [1, BN] }> = offs_bn.reshape(const_shape![1, BN]);
                    let obn_2d: Tile<i32, { [BK, BN] }> = obn_row.broadcast(const_shape![BK, BN]);
                    let b_off_a: Tile<i32, { [BK, BN] }> = be_2d + ok_n;
                    let b_off: Tile<i32, { [BK, BN] }> = b_off_a + obn_2d;
                    let b_base: Tile<i64, { [BK, BN] }> = broadcast_scalar(b_base_s, const_shape![BK, BN]);
                    let b_off64: Tile<i64, { [BK, BN] }> = exti(b_off);
                    let eb_b: Tile<i64, { [BK, BN] }> = broadcast_scalar(elem_bytes, const_shape![BK, BN]);
                    let b_byte: Tile<i64, { [BK, BN] }> = muli(b_off64, eb_b, overflow::NoSignedWrap);
                    let b_addr: Tile<i64, { [BK, BN] }> = b_base + b_byte;
                    let b_ptrs: PointerTile<*mut T, { [BK, BN] }> = int_to_ptr(b_addr);
                    let (b_tile, _): (Tile<T, { [BK, BN] }>, Token) = load_ptr_tko(
                        b_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );

                    acc = mmaf(a_tile, b_tile, acc);
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
                    let moe_w_2d: Tile<f32, { [BM, BN] }> = moe_w_col.broadcast(const_shape![BM, BN]);
                    acc = acc * moe_w_2d;
                }

                let acc_bf: Tile<T, { [BM, BN] }> = convert_tile(acc);
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
