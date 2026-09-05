#![allow(
    clippy::too_many_arguments,
    reason = "cuTile kernel arguments follow the device ABI"
)]

#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::cutile_compiler;

    const BLOCK: i32 = 16;
    const FP4_MAX: f32 = 6.0;
    const FP4_MIN: f32 = -6.0;
    const FP8_MAX: f32 = 448.0;

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn quantize_bf16<const BM: i32, const BK: i32, const PK: i32, const SK: i32>(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        rows: i32,
        scale_stride: i32,
        x: &Tensor<bf16, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            let xt: Tile<bf16, { [BM, BK] }> = px.load([m, kg]);
            let xf: Tile<f32, { [BM, BK] }> = convert_tile(xt);
            let xf = xf
                / global
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![BM, BK]);
            let blocks: Tile<f32, { [BM, SK, BLOCK] }> = xf.reshape(const_shape![BM, SK, BLOCK]);
            let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
            let maxima = maxima.reshape(const_shape![BM, SK]);
            let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
            let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
            let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
            let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
            let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
            let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
            let raw_scale = min_tile(maxima / fp4_max, fp8_max);
            let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
            let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
            let denominator = select(eq_tile(rounded, zeros), ones, rounded);
            let denominator = denominator
                .reshape(const_shape![BM, SK, 1])
                .broadcast(const_shape![BM, SK, BLOCK]);
            let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
            let normalized = max_tile(min_tile(normalized, upper), lower);
            let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
            let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
            q.store(packed, idx);
            let row: Tile<i32, { [BM] }> =
                iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
            let col: Tile<i32, { [SK] }> =
                iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
            let row_offset: Tile<i64, { [BM] }> = exti(row);
            let stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(scale_stride, const_shape![BM]));
            let col_offset: Tile<i64, { [SK] }> = exti(col);
            let offset = (row_offset * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                + col_offset
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
            let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> = base.reshape(const_shape![1, 1]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                base.broadcast(const_shape![BM, SK]);
            let ptrs: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
            store_ptr_tko(
                ptrs,
                sx,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    fn matmul_bf16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<bf16, { [BM, BN] }, { [1, 1] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        qs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1] }>,
        wg: &Tensor<f32, { [-1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pq = q.partition(const_shape![BM, PK]);
        let pqs = qs.partition(const_shape![BM, SK]);
        let pw = w.partition(const_shape![BN, PK]);
        let pws = ws.partition(const_shape![BN, SK]);
        let pwg = wg.partition(const_shape![BN]);
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let k = num_tiles(&pw, 1);
        for idx in y.iter_indices() {
            let (m, n) = idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> = pw.load_pipelined::<LATENCY>([n, kg]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> = pws.load_pipelined::<LATENCY>([n, kg]);
                if A4 {
                    let xp: Tile<f4e2m1fnx2, { [BM, PK] }> = pq.load_pipelined::<LATENCY>([m, kg]);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = xp.unpack(const_shape![BM, BK]);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = pqs.load_pipelined::<LATENCY>([m, kg]);
                    let wt: Tile<f4e2m1fn, { [BK, BN] }> = permute(wt, transpose);
                    let sw: Tile<f8e4m3fn, { [SK, BN] }> = permute(sw, transpose);
                    acc = mmaf_scaled(xq, wt, acc, sx, sw);
                } else {
                    let xt: Tile<bf16, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, kg]);
                    let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                    let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                    let sw = sw
                        .reshape(const_shape![BN, SK, 1])
                        .broadcast(const_shape![BN, SK, BLOCK])
                        .reshape(const_shape![BN, BK]);
                    let wd: Tile<bf16, { [BN, BK] }> = convert_tile(wf * sw);
                    let wd: Tile<bf16, { [BK, BN] }> = permute(wd, transpose);
                    acc = mmaf(xt, wd, acc);
                }
            }
            let weight_global: Tile<f32, { [BN] }> = pwg.load([n]);
            let weight_global = weight_global
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let result = acc * weight_global;
            let result = if A4 {
                result
                    * global
                        .reshape(const_shape![1, 1])
                        .broadcast(const_shape![BM, BN])
            } else {
                result
            };
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(result);
            y.store(out, idx);
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn quantize_f16<const BM: i32, const BK: i32, const PK: i32, const SK: i32>(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        rows: i32,
        scale_stride: i32,
        x: &Tensor<f16, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            let xt: Tile<f16, { [BM, BK] }> = px.load([m, kg]);
            let xf: Tile<f32, { [BM, BK] }> = convert_tile(xt);
            let xf = xf
                / global
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![BM, BK]);
            let blocks: Tile<f32, { [BM, SK, BLOCK] }> = xf.reshape(const_shape![BM, SK, BLOCK]);
            let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
            let maxima = maxima.reshape(const_shape![BM, SK]);
            let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
            let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
            let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
            let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
            let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
            let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
            let raw_scale = min_tile(maxima / fp4_max, fp8_max);
            let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
            let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
            let denominator = select(eq_tile(rounded, zeros), ones, rounded);
            let denominator = denominator
                .reshape(const_shape![BM, SK, 1])
                .broadcast(const_shape![BM, SK, BLOCK]);
            let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
            let normalized = max_tile(min_tile(normalized, upper), lower);
            let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
            let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
            q.store(packed, idx);
            let row: Tile<i32, { [BM] }> =
                iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
            let col: Tile<i32, { [SK] }> =
                iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
            let row_offset: Tile<i64, { [BM] }> = exti(row);
            let stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(scale_stride, const_shape![BM]));
            let col_offset: Tile<i64, { [SK] }> = exti(col);
            let offset = (row_offset * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                + col_offset
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
            let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> = base.reshape(const_shape![1, 1]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                base.broadcast(const_shape![BM, SK]);
            let ptrs: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
            store_ptr_tko(
                ptrs,
                sx,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    fn matmul_f16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<f16, { [BM, BN] }, { [1, 1] }>,
        x: &Tensor<f16, { [-1, -1] }>,
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        qs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1] }>,
        wg: &Tensor<f32, { [-1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pq = q.partition(const_shape![BM, PK]);
        let pqs = qs.partition(const_shape![BM, SK]);
        let pw = w.partition(const_shape![BN, PK]);
        let pws = ws.partition(const_shape![BN, SK]);
        let pwg = wg.partition(const_shape![BN]);
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let k = num_tiles(&pw, 1);
        for idx in y.iter_indices() {
            let (m, n) = idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> = pw.load_pipelined::<LATENCY>([n, kg]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> = pws.load_pipelined::<LATENCY>([n, kg]);
                if A4 {
                    let xp: Tile<f4e2m1fnx2, { [BM, PK] }> = pq.load_pipelined::<LATENCY>([m, kg]);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = xp.unpack(const_shape![BM, BK]);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = pqs.load_pipelined::<LATENCY>([m, kg]);
                    let wt: Tile<f4e2m1fn, { [BK, BN] }> = permute(wt, transpose);
                    let sw: Tile<f8e4m3fn, { [SK, BN] }> = permute(sw, transpose);
                    acc = mmaf_scaled(xq, wt, acc, sx, sw);
                } else {
                    let xt: Tile<f16, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, kg]);
                    let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                    let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                    let sw = sw
                        .reshape(const_shape![BN, SK, 1])
                        .broadcast(const_shape![BN, SK, BLOCK])
                        .reshape(const_shape![BN, BK]);
                    let wd: Tile<f16, { [BN, BK] }> = convert_tile(wf * sw);
                    let wd: Tile<f16, { [BK, BN] }> = permute(wd, transpose);
                    acc = mmaf(xt, wd, acc);
                }
            }
            let weight_global: Tile<f32, { [BN] }> = pwg.load([n]);
            let weight_global = weight_global
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let result = acc * weight_global;
            let result = if A4 {
                result
                    * global
                        .reshape(const_shape![1, 1])
                        .broadcast(const_shape![BM, BN])
            } else {
                result
            };
            let out: Tile<f16, { [BM, BN] }> = convert_tile(result);
            y.store(out, idx);
        }
    }
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn route_quantize_bf16<
        const BM: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const ALIGN_M: i32,
        const A4: bool,
    >(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        rows: i32,
        scale_stride: i32,
        x: *mut bf16,
        x_out: *mut bf16,
        sids: &Tensor<i32, { [-1] }>,
        eids: &Tensor<i32, { [-1] }>,
        ntpp: &Tensor<i32, { [-1] }>,
        routes: i32,
        x_stride: i32,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let aligned: Tile<i32, { [1] }> = load_tile(ntpp, const_shape![1], [0]);
        let aligned: i32 = tile_to_scalar(aligned.reshape(const_shape![]));
        let pi = sids.partition(const_shape![BM]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            if m * BM < aligned {
                let expert: Tile<i32, { [1] }> =
                    load_tile(eids, const_shape![1], [m * BM / ALIGN_M]);
                let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
                let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [expert]);
                let ids: Tile<i32, { [BM] }> = pi.load([m]);
                let valid: Tile<bool, { [BM] }> =
                    lt_tile(ids, broadcast_scalar(routes, const_shape![BM]));
                let cols: Tile<i32, { [BK] }> =
                    iota(const_shape![BK]) + broadcast_scalar(kg * BK, const_shape![BK]);
                let ksize: i32 = scale_stride * BLOCK;
                let in_rows: Tile<i32, { [BM] }> =
                    ids / broadcast_scalar(x_stride, const_shape![BM]);
                let row_offset: Tile<i64, { [BM] }> = exti(in_rows);
                let stride: Tile<i64, { [BM] }> = exti(broadcast_scalar(ksize, const_shape![BM]));
                let col_offset: Tile<i64, { [BK] }> = exti(cols);
                let offsets = (row_offset * stride)
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BK])
                    + col_offset
                        .reshape(const_shape![1, BK])
                        .broadcast(const_shape![BM, BK]);
                let input_mask = valid
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BK])
                    & lt_tile(cols, broadcast_scalar(ksize, const_shape![BK]))
                        .reshape(const_shape![1, BK])
                        .broadcast(const_shape![BM, BK]);
                let ptr: PointerTile<*mut bf16, { [] }> = pointer_to_tile(x);
                let ptr: PointerTile<*mut bf16, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
                let ptr: PointerTile<*mut bf16, { [BM, BK] }> = ptr.broadcast(const_shape![BM, BK]);
                let ptr: PointerTile<*mut bf16, { [BM, BK] }> = ptr.offset_tile(offsets);
                let (xt, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                    ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(input_mask),
                    None,
                    None,
                    Latency::<0>,
                );
                let zeros: Tile<f32, { [BM, BK] }> = constant(0.0f32, const_shape![BM, BK]);
                let zeros: Tile<bf16, { [BM, BK] }> = convert_tile(zeros);
                let xt: Tile<bf16, { [BM, BK] }> = select(input_mask, xt, zeros);
                if A4 {
                    let xf: Tile<f32, { [BM, BK] }> = convert_tile(xt);
                    let xf = xf
                        / global
                            .reshape(const_shape![1, 1])
                            .broadcast(const_shape![BM, BK]);
                    let blocks: Tile<f32, { [BM, SK, BLOCK] }> =
                        xf.reshape(const_shape![BM, SK, BLOCK]);
                    let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
                    let maxima = maxima.reshape(const_shape![BM, SK]);
                    let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
                    let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
                    let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
                    let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
                    let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
                    let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
                    let raw_scale = min_tile(maxima / fp4_max, fp8_max);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
                    let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
                    let denominator = select(eq_tile(rounded, zeros), ones, rounded);
                    let denominator = denominator
                        .reshape(const_shape![BM, SK, 1])
                        .broadcast(const_shape![BM, SK, BLOCK]);
                    let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
                    let normalized = max_tile(min_tile(normalized, upper), lower);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
                    let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
                    q.store(packed, idx);
                    let row: Tile<i32, { [BM] }> =
                        iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
                    let col: Tile<i32, { [SK] }> =
                        iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
                    let row_offset: Tile<i64, { [BM] }> = exti(row);
                    let stride: Tile<i64, { [BM] }> =
                        exti(broadcast_scalar(scale_stride, const_shape![BM]));
                    let col_offset: Tile<i64, { [SK] }> = exti(col);
                    let offset = (row_offset * stride)
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, SK])
                        + col_offset
                            .reshape(const_shape![1, SK])
                            .broadcast(const_shape![BM, SK]);
                    let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, SK])
                        & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                            .reshape(const_shape![1, SK])
                            .broadcast(const_shape![BM, SK]);
                    let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
                    let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> =
                        base.reshape(const_shape![1, 1]);
                    let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                        base.broadcast(const_shape![BM, SK]);
                    let ptrs: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
                    store_ptr_tko(
                        ptrs,
                        sx,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None,
                        Latency::<0>,
                    );
                } else {
                    let out_rows: Tile<i32, { [BM] }> =
                        iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
                    let row_offset: Tile<i64, { [BM] }> = exti(out_rows);
                    let stride: Tile<i64, { [BM] }> =
                        exti(broadcast_scalar(ksize, const_shape![BM]));
                    let col_offset: Tile<i64, { [BK] }> = exti(cols);
                    let offsets = (row_offset * stride)
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BK])
                        + col_offset
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BM, BK]);
                    let mask = lt_tile(out_rows, broadcast_scalar(rows, const_shape![BM]))
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BK])
                        & lt_tile(cols, broadcast_scalar(ksize, const_shape![BK]))
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BM, BK]);
                    let ptr: PointerTile<*mut bf16, { [] }> = pointer_to_tile(x_out);
                    let ptr: PointerTile<*mut bf16, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
                    let ptr: PointerTile<*mut bf16, { [BM, BK] }> =
                        ptr.broadcast(const_shape![BM, BK]);
                    let ptr: PointerTile<*mut bf16, { [BM, BK] }> = ptr.offset_tile(offsets);
                    store_ptr_tko(
                        ptr,
                        xt,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None,
                        Latency::<0>,
                    );
                }
            }
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn routed_matmul_bf16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const LATENCY: i32,
    >(
        y: *mut bf16,
        sids: &Tensor<i32, { [-1] }>,
        eids: &Tensor<i32, { [-1] }>,
        ntpp: &Tensor<i32, { [-1] }>,
        routes: i32,
        n_size: i32,
        x: &Tensor<bf16, { [-1, -1] }>,
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        qs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        wg: &Tensor<f32, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pq = q.partition(const_shape![BM, PK]);
        let pqs = qs.partition(const_shape![BM, SK]);
        let pw = w.partition(const_shape![1, BN, PK]);
        let pws = ws.partition(const_shape![1, BN, SK]);
        let pwg = wg.partition(const_shape![1, BN]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let k = num_tiles(&pw, 2);
        let pid = get_tile_block_id().0;
        let ntiles = ceil_div(n_size, BN);
        let m = pid / ntiles;
        let n = pid % ntiles;
        let aligned: Tile<i32, { [1] }> = load_tile(ntpp, const_shape![1], [0]);
        let aligned: i32 = tile_to_scalar(aligned.reshape(const_shape![]));
        if m * BM < aligned {
            let expert: Tile<i32, { [1] }> = load_tile(eids, const_shape![1], [m]);
            let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
            let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [expert]);
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> = pw
                    .load_pipelined::<LATENCY>([expert, n, kg])
                    .reshape(const_shape![BN, PK]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> = pws
                    .load_pipelined::<LATENCY>([expert, n, kg])
                    .reshape(const_shape![BN, SK]);
                if A4 {
                    let xp: Tile<f4e2m1fnx2, { [BM, PK] }> = pq.load_pipelined::<LATENCY>([m, kg]);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = xp.unpack(const_shape![BM, BK]);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = pqs.load_pipelined::<LATENCY>([m, kg]);
                    let wt: Tile<f4e2m1fn, { [BK, BN] }> = permute(wt, transpose);
                    let sw: Tile<f8e4m3fn, { [SK, BN] }> = permute(sw, transpose);
                    acc = mmaf_scaled(xq, wt, acc, sx, sw);
                } else {
                    let xt: Tile<bf16, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, kg]);
                    let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                    let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                    let sw = sw
                        .reshape(const_shape![BN, SK, 1])
                        .broadcast(const_shape![BN, SK, BLOCK])
                        .reshape(const_shape![BN, BK]);
                    let wd: Tile<bf16, { [BN, BK] }> = convert_tile(wf * sw);
                    let wd: Tile<bf16, { [BK, BN] }> = permute(wd, transpose);
                    acc = mmaf(xt, wd, acc);
                }
            }
            let weight_global: Tile<f32, { [BN] }> =
                pwg.load([expert, n]).reshape(const_shape![BN]);
            let weight_global = weight_global
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let result = acc * weight_global;
            let result = if A4 {
                result
                    * global
                        .reshape(const_shape![1, 1])
                        .broadcast(const_shape![BM, BN])
            } else {
                result
            };
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(result);
            let ids: Tile<i32, { [BM] }> = load_tile(sids, const_shape![BM], [m]);
            let cols: Tile<i32, { [BN] }> =
                iota(const_shape![BN]) + broadcast_scalar(n * BN, const_shape![BN]);
            let row_offset: Tile<i64, { [BM] }> = exti(ids);
            let stride: Tile<i64, { [BM] }> = exti(broadcast_scalar(n_size, const_shape![BM]));
            let col_offset: Tile<i64, { [BN] }> = exti(cols);
            let offsets = (row_offset * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN])
                + col_offset
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
            let mask = lt_tile(ids, broadcast_scalar(routes, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN])
                & lt_tile(cols, broadcast_scalar(n_size, const_shape![BN]))
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
            let ptr: PointerTile<*mut bf16, { [] }> = pointer_to_tile(y);
            let ptr: PointerTile<*mut bf16, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
            let ptr: PointerTile<*mut bf16, { [BM, BN] }> = ptr.broadcast(const_shape![BM, BN]);
            let ptr: PointerTile<*mut bf16, { [BM, BN] }> = ptr.offset_tile(offsets);
            store_ptr_tko(
                ptr,
                out,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn route_quantize_f16<
        const BM: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const ALIGN_M: i32,
        const A4: bool,
    >(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        rows: i32,
        scale_stride: i32,
        x: *mut f16,
        x_out: *mut f16,
        sids: &Tensor<i32, { [-1] }>,
        eids: &Tensor<i32, { [-1] }>,
        ntpp: &Tensor<i32, { [-1] }>,
        routes: i32,
        x_stride: i32,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let aligned: Tile<i32, { [1] }> = load_tile(ntpp, const_shape![1], [0]);
        let aligned: i32 = tile_to_scalar(aligned.reshape(const_shape![]));
        let pi = sids.partition(const_shape![BM]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            if m * BM < aligned {
                let expert: Tile<i32, { [1] }> =
                    load_tile(eids, const_shape![1], [m * BM / ALIGN_M]);
                let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
                let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [expert]);
                let ids: Tile<i32, { [BM] }> = pi.load([m]);
                let valid: Tile<bool, { [BM] }> =
                    lt_tile(ids, broadcast_scalar(routes, const_shape![BM]));
                let cols: Tile<i32, { [BK] }> =
                    iota(const_shape![BK]) + broadcast_scalar(kg * BK, const_shape![BK]);
                let ksize: i32 = scale_stride * BLOCK;
                let in_rows: Tile<i32, { [BM] }> =
                    ids / broadcast_scalar(x_stride, const_shape![BM]);
                let row_offset: Tile<i64, { [BM] }> = exti(in_rows);
                let stride: Tile<i64, { [BM] }> = exti(broadcast_scalar(ksize, const_shape![BM]));
                let col_offset: Tile<i64, { [BK] }> = exti(cols);
                let offsets = (row_offset * stride)
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BK])
                    + col_offset
                        .reshape(const_shape![1, BK])
                        .broadcast(const_shape![BM, BK]);
                let input_mask = valid
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BK])
                    & lt_tile(cols, broadcast_scalar(ksize, const_shape![BK]))
                        .reshape(const_shape![1, BK])
                        .broadcast(const_shape![BM, BK]);
                let ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(x);
                let ptr: PointerTile<*mut f16, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
                let ptr: PointerTile<*mut f16, { [BM, BK] }> = ptr.broadcast(const_shape![BM, BK]);
                let ptr: PointerTile<*mut f16, { [BM, BK] }> = ptr.offset_tile(offsets);
                let (xt, _): (Tile<f16, { [BM, BK] }>, Token) = load_ptr_tko(
                    ptr,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(input_mask),
                    None,
                    None,
                    Latency::<0>,
                );
                let zeros: Tile<f32, { [BM, BK] }> = constant(0.0f32, const_shape![BM, BK]);
                let zeros: Tile<f16, { [BM, BK] }> = convert_tile(zeros);
                let xt: Tile<f16, { [BM, BK] }> = select(input_mask, xt, zeros);
                if A4 {
                    let xf: Tile<f32, { [BM, BK] }> = convert_tile(xt);
                    let xf = xf
                        / global
                            .reshape(const_shape![1, 1])
                            .broadcast(const_shape![BM, BK]);
                    let blocks: Tile<f32, { [BM, SK, BLOCK] }> =
                        xf.reshape(const_shape![BM, SK, BLOCK]);
                    let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
                    let maxima = maxima.reshape(const_shape![BM, SK]);
                    let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
                    let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
                    let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
                    let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
                    let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
                    let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
                    let raw_scale = min_tile(maxima / fp4_max, fp8_max);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
                    let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
                    let denominator = select(eq_tile(rounded, zeros), ones, rounded);
                    let denominator = denominator
                        .reshape(const_shape![BM, SK, 1])
                        .broadcast(const_shape![BM, SK, BLOCK]);
                    let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
                    let normalized = max_tile(min_tile(normalized, upper), lower);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
                    let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
                    q.store(packed, idx);
                    let row: Tile<i32, { [BM] }> =
                        iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
                    let col: Tile<i32, { [SK] }> =
                        iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
                    let row_offset: Tile<i64, { [BM] }> = exti(row);
                    let stride: Tile<i64, { [BM] }> =
                        exti(broadcast_scalar(scale_stride, const_shape![BM]));
                    let col_offset: Tile<i64, { [SK] }> = exti(col);
                    let offset = (row_offset * stride)
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, SK])
                        + col_offset
                            .reshape(const_shape![1, SK])
                            .broadcast(const_shape![BM, SK]);
                    let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, SK])
                        & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                            .reshape(const_shape![1, SK])
                            .broadcast(const_shape![BM, SK]);
                    let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
                    let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> =
                        base.reshape(const_shape![1, 1]);
                    let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                        base.broadcast(const_shape![BM, SK]);
                    let ptrs: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
                    store_ptr_tko(
                        ptrs,
                        sx,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None,
                        Latency::<0>,
                    );
                } else {
                    let out_rows: Tile<i32, { [BM] }> =
                        iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
                    let row_offset: Tile<i64, { [BM] }> = exti(out_rows);
                    let stride: Tile<i64, { [BM] }> =
                        exti(broadcast_scalar(ksize, const_shape![BM]));
                    let col_offset: Tile<i64, { [BK] }> = exti(cols);
                    let offsets = (row_offset * stride)
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BK])
                        + col_offset
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BM, BK]);
                    let mask = lt_tile(out_rows, broadcast_scalar(rows, const_shape![BM]))
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BK])
                        & lt_tile(cols, broadcast_scalar(ksize, const_shape![BK]))
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BM, BK]);
                    let ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(x_out);
                    let ptr: PointerTile<*mut f16, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
                    let ptr: PointerTile<*mut f16, { [BM, BK] }> =
                        ptr.broadcast(const_shape![BM, BK]);
                    let ptr: PointerTile<*mut f16, { [BM, BK] }> = ptr.offset_tile(offsets);
                    store_ptr_tko(
                        ptr,
                        xt,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None,
                        Latency::<0>,
                    );
                }
            }
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn routed_matmul_f16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const LATENCY: i32,
    >(
        y: *mut f16,
        sids: &Tensor<i32, { [-1] }>,
        eids: &Tensor<i32, { [-1] }>,
        ntpp: &Tensor<i32, { [-1] }>,
        routes: i32,
        n_size: i32,
        x: &Tensor<f16, { [-1, -1] }>,
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        qs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        wg: &Tensor<f32, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pq = q.partition(const_shape![BM, PK]);
        let pqs = qs.partition(const_shape![BM, SK]);
        let pw = w.partition(const_shape![1, BN, PK]);
        let pws = ws.partition(const_shape![1, BN, SK]);
        let pwg = wg.partition(const_shape![1, BN]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let k = num_tiles(&pw, 2);
        let pid = get_tile_block_id().0;
        let ntiles = ceil_div(n_size, BN);
        let m = pid / ntiles;
        let n = pid % ntiles;
        let aligned: Tile<i32, { [1] }> = load_tile(ntpp, const_shape![1], [0]);
        let aligned: i32 = tile_to_scalar(aligned.reshape(const_shape![]));
        if m * BM < aligned {
            let expert: Tile<i32, { [1] }> = load_tile(eids, const_shape![1], [m]);
            let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
            let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [expert]);
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> = pw
                    .load_pipelined::<LATENCY>([expert, n, kg])
                    .reshape(const_shape![BN, PK]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> = pws
                    .load_pipelined::<LATENCY>([expert, n, kg])
                    .reshape(const_shape![BN, SK]);
                if A4 {
                    let xp: Tile<f4e2m1fnx2, { [BM, PK] }> = pq.load_pipelined::<LATENCY>([m, kg]);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = xp.unpack(const_shape![BM, BK]);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = pqs.load_pipelined::<LATENCY>([m, kg]);
                    let wt: Tile<f4e2m1fn, { [BK, BN] }> = permute(wt, transpose);
                    let sw: Tile<f8e4m3fn, { [SK, BN] }> = permute(sw, transpose);
                    acc = mmaf_scaled(xq, wt, acc, sx, sw);
                } else {
                    let xt: Tile<f16, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, kg]);
                    let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                    let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                    let sw = sw
                        .reshape(const_shape![BN, SK, 1])
                        .broadcast(const_shape![BN, SK, BLOCK])
                        .reshape(const_shape![BN, BK]);
                    let wd: Tile<f16, { [BN, BK] }> = convert_tile(wf * sw);
                    let wd: Tile<f16, { [BK, BN] }> = permute(wd, transpose);
                    acc = mmaf(xt, wd, acc);
                }
            }
            let weight_global: Tile<f32, { [BN] }> =
                pwg.load([expert, n]).reshape(const_shape![BN]);
            let weight_global = weight_global
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let result = acc * weight_global;
            let result = if A4 {
                result
                    * global
                        .reshape(const_shape![1, 1])
                        .broadcast(const_shape![BM, BN])
            } else {
                result
            };
            let out: Tile<f16, { [BM, BN] }> = convert_tile(result);
            let ids: Tile<i32, { [BM] }> = load_tile(sids, const_shape![BM], [m]);
            let cols: Tile<i32, { [BN] }> =
                iota(const_shape![BN]) + broadcast_scalar(n * BN, const_shape![BN]);
            let row_offset: Tile<i64, { [BM] }> = exti(ids);
            let stride: Tile<i64, { [BM] }> = exti(broadcast_scalar(n_size, const_shape![BM]));
            let col_offset: Tile<i64, { [BN] }> = exti(cols);
            let offsets = (row_offset * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN])
                + col_offset
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
            let mask = lt_tile(ids, broadcast_scalar(routes, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN])
                & lt_tile(cols, broadcast_scalar(n_size, const_shape![BN]))
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
            let ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(y);
            let ptr: PointerTile<*mut f16, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
            let ptr: PointerTile<*mut f16, { [BM, BN] }> = ptr.broadcast(const_shape![BM, BN]);
            let ptr: PointerTile<*mut f16, { [BM, BN] }> = ptr.offset_tile(offsets);
            store_ptr_tko(
                ptr,
                out,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
}

use std::sync::Arc;

use candle_core::{CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
use cutile::core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;
use half::{bf16, f16};

use super::nvfp4::Nvfp4GemmArgs;
use super::{catch_cutile_panic, context, device_multiprocessor_count};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const BLOCK_SIZE: usize = 16;
const QUANT_ROWS: usize = 4;
const QUANT_K: usize = 256;
const SMALL_MATMUL_ROWS: usize = 16;
const SMALL_MATMUL_COLUMNS: usize = 64;
const MATMUL_ROWS: usize = 64;
const MATMUL_COLUMNS: usize = 128;
const MATMUL_K: usize = 256;
const WIDE_MATMUL_ROWS: usize = 128;
const WIDE_MATMUL_COLUMNS: usize = 64;
const WIDE_MATMUL_MIN_K: usize = 8192;
const WIDE_MATMUL_MIN_N: usize = 4096;
const LOAD_LATENCY: usize = 3;
const BLOCKS_PER_SM: usize = 2;

pub(super) fn launch(x: &Tensor, args: Nvfp4GemmArgs<'_>, compile_only: bool) -> Result<Tensor> {
    let (m, k) = x.dims2()?;
    let (n, packed_k) = args.weights.dims2()?;
    let (bm, bn) = if m <= SMALL_MATMUL_ROWS {
        (SMALL_MATMUL_ROWS, SMALL_MATMUL_COLUMNS)
    } else if m >= WIDE_MATMUL_ROWS && n >= WIDE_MATMUL_MIN_N && k >= WIDE_MATMUL_MIN_K {
        (WIDE_MATMUL_ROWS, WIDE_MATMUL_COLUMNS)
    } else {
        (MATMUL_ROWS, MATMUL_COLUMNS)
    };
    let Device::Cuda(dev) = x.device() else {
        candle_core::bail!("cuTile NVFP4 matmul requires CUDA tensors")
    };
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let cutile_stream = context::stream(dev);
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = args.weights.storage_and_layout();
    let (s_storage, s_layout) = args.weight_scales.storage_and_layout();
    let (wg_storage, wg_layout) = args.weight_global_scale.storage_and_layout();
    let activation_global = args
        .activation_global_scale
        .unwrap_or(args.weight_global_scale);
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (
        Storage::Cuda(x_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(s_cuda),
        Storage::Cuda(wg_cuda),
        Storage::Cuda(ag_cuda),
    ) = (
        &*x_storage,
        &*w_storage,
        &*s_storage,
        &*wg_storage,
        &*ag_storage,
    )
    else {
        candle_core::bail!("cuTile NVFP4 matmul operands must be CUDA tensors")
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (wg_addr, _wg_guard) = slice_ptr_on_stream(
        wg_cuda.as_cuda_slice::<f32>()?,
        wg_layout.start_offset(),
        &stream,
    );
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let w = unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            vec![n as i32, packed_k as i32],
            vec![packed_k as i32, 1],
        )
    };
    let ws = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            vec![n as i32, (k / BLOCK_SIZE) as i32],
            vec![(k / BLOCK_SIZE) as i32, 1],
        )
    };
    let wg = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            wg_addr as CUdeviceptr,
            ordinal,
            vec![n as i32],
            vec![1],
        )
    };
    let ag = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ag_addr as CUdeviceptr,
            ordinal,
            vec![1],
            vec![1],
        )
    };
    let blocks = (BLOCKS_PER_SM * device_multiprocessor_count(dev)) as u32;
    let a4 = args.activation_global_scale.is_some();
    let generic = vec![
        bm.to_string(),
        bn.to_string(),
        MATMUL_K.to_string(),
        (MATMUL_K / 2).to_string(),
        (MATMUL_K / BLOCK_SIZE).to_string(),
        a4.to_string(),
        LOAD_LATENCY.to_string(),
    ];
    let quant_generic = vec![
        QUANT_ROWS.to_string(),
        QUANT_K.to_string(),
        (QUANT_K / 2).to_string(),
        (QUANT_K / BLOCK_SIZE).to_string(),
    ];

    macro_rules! dispatch {
        ($launcher:expr, $generic:expr) => {{
            let launcher = $launcher.generics($generic);
            if compile_only {
                catch_cutile_panic("NVFP4 matmul compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile NVFP4 matmul compile failed: {error:?}"
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("NVFP4 matmul launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile NVFP4 matmul launch failed: {error:?}"
                        ))
                    })
                })?;
            }
        }};
    }
    macro_rules! run {
        ($dtype:ty, $quant:path, $matmul:path) => {{
            let (x_addr, _x_guard) = slice_ptr_on_stream(
                x_cuda.as_cuda_slice::<$dtype>()?,
                x_layout.start_offset(),
                &stream,
            );
            let x = Arc::new(unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    x_addr as CUdeviceptr,
                    ordinal,
                    vec![m as i32, k as i32],
                    vec![k as i32, 1],
                )
            });
            let ag = Arc::new(ag);
            let mut output = unsafe { dev.alloc::<$dtype>(m * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let y = unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    out_addr as CUdeviceptr,
                    ordinal,
                    vec![m as i32, n as i32],
                    vec![n as i32, 1],
                )
            };
            let mut packed = unsafe { dev.alloc::<u8>(if a4 { m * packed_k } else { 1 })? };
            let mut scales =
                unsafe { dev.alloc::<F8E4M3>(if a4 { m * k / BLOCK_SIZE } else { 1 })? };
            let (q_addr, q_guard) = slice_ptr_mut_on_stream(&mut packed, 0, &stream);
            let (qs_addr, qs_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
            let q = unsafe {
                cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
                    q_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![m as i32, packed_k as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![packed_k as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            };
            let qs = unsafe {
                cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
                    qs_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![m as i32, (k / BLOCK_SIZE) as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![(k / BLOCK_SIZE) as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            };
            let q_read = Arc::new(unsafe {
                cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
                    q_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![m as i32, packed_k as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![packed_k as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            });
            let qs_read = Arc::new(qs);
            if a4 {
                let quant = q.partition([QUANT_ROWS, QUANT_K / 2]).map(
                    [1, 1],
                    blocks.min((m.div_ceil(QUANT_ROWS) * k.div_ceil(QUANT_K)) as u32),
                );
                dispatch!(
                    unsafe {
                        $quant(
                            quant,
                            DevicePointer::<f8e4m3fn>::from_cu_deviceptr(qs_addr as CUdeviceptr),
                            m as i32,
                            (k / BLOCK_SIZE) as i32,
                            x.clone(),
                            ag.clone(),
                        )
                    },
                    quant_generic
                );
            }
            let mapped = y
                .partition([bm, bn])
                .map([1, 1], blocks.min((m.div_ceil(bm) * n.div_ceil(bn)) as u32));
            dispatch!(
                $matmul(
                    mapped,
                    x,
                    q_read,
                    qs_read,
                    Arc::new(w),
                    Arc::new(ws),
                    Arc::new(wg),
                    ag
                ),
                generic
            );
            drop(q_guard);
            drop(qs_guard);
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[m, n]),
            ))
        }};
    }
    match x.dtype() {
        DType::BF16 => Ok(run!(bf16, kernels::quantize_bf16, kernels::matmul_bf16)),
        DType::F16 => Ok(run!(f16, kernels::quantize_f16, kernels::matmul_f16)),
        dtype => candle_core::bail!("cuTile NVFP4 matmul does not support {dtype:?} activations"),
    }
}

pub(super) const ROUTED_ROWS: usize = 16;
const ROUTED_COLUMNS: usize = 64;

pub(super) fn launch_gather(
    x: &Tensor,
    indices: &Tensor,
    args: Nvfp4GemmArgs<'_>,
    compile_only: bool,
) -> Result<Tensor> {
    let (tokens, topk) = indices.dims2()?;
    let (experts, n, packed_k) = args.weights.dims3()?;
    let routes = tokens * topk;
    let k = packed_k * 2;
    let input_rows = x.elem_count() / k;
    let x_stride = routes / input_rows;
    let x = x.contiguous()?.reshape((input_rows, k))?;
    let Device::Cuda(dev) = x.device() else {
        candle_core::bail!("cuTile NVFP4 grouped matmul requires CUDA tensors")
    };
    let indices = indices.contiguous()?;
    let indices = if indices.layout().start_offset() == 0 {
        indices
    } else {
        indices.copy()?
    };
    let (id_storage, _) = indices.storage_and_layout();
    let Storage::Cuda(id_cuda) = &*id_storage else {
        candle_core::bail!("cuTile NVFP4 grouped indices must be CUDA tensors")
    };
    let alignment = if compile_only {
        let em = crate::moe::cuda::moe_align_em(tokens, topk, experts, ROUTED_ROWS);
        super::fused_moe::MoeAlign {
            sids: unsafe { dev.alloc::<i32>(em)? },
            eids: unsafe { dev.alloc::<i32>(em.div_ceil(ROUTED_ROWS))? },
            ntpp: unsafe { dev.alloc::<i32>(1)? },
            em,
        }
    } else {
        super::fused_moe::MoeAlign::build(
            dev,
            id_cuda.as_cuda_slice::<u32>()?,
            tokens,
            experts,
            topk,
            ROUTED_ROWS as i32,
        )?
    };
    let em = alignment.em;
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let cutile_stream = context::stream(dev);
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = args.weights.storage_and_layout();
    let (s_storage, s_layout) = args.weight_scales.storage_and_layout();
    let (wg_storage, wg_layout) = args.weight_global_scale.storage_and_layout();
    let activation_global = args
        .activation_global_scale
        .unwrap_or(args.weight_global_scale);
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (
        Storage::Cuda(x_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(s_cuda),
        Storage::Cuda(wg_cuda),
        Storage::Cuda(ag_cuda),
    ) = (
        &*x_storage,
        &*w_storage,
        &*s_storage,
        &*wg_storage,
        &*ag_storage,
    )
    else {
        candle_core::bail!("cuTile NVFP4 grouped matmul operands must be CUDA tensors")
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (wg_addr, _wg_guard) = slice_ptr_on_stream(
        wg_cuda.as_cuda_slice::<f32>()?,
        wg_layout.start_offset(),
        &stream,
    );
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let (sid_addr, _sid_guard) = slice_ptr_on_stream(&alignment.sids, 0, &stream);
    let (eid_addr, _eid_guard) = slice_ptr_on_stream(&alignment.eids, 0, &stream);
    let (ntpp_addr, _ntpp_guard) = slice_ptr_on_stream(&alignment.ntpp, 0, &stream);
    let sids = Arc::new(unsafe {
        cutile::tensor::Tensor::<i32>::borrow_raw_parts(
            sid_addr as CUdeviceptr,
            ordinal,
            vec![em as i32],
            vec![1],
        )
    });
    let eids = Arc::new(unsafe {
        cutile::tensor::Tensor::<i32>::borrow_raw_parts(
            eid_addr as CUdeviceptr,
            ordinal,
            vec![em.div_ceil(ROUTED_ROWS) as i32],
            vec![1],
        )
    });
    let ntpp = Arc::new(unsafe {
        cutile::tensor::Tensor::<i32>::borrow_raw_parts(
            ntpp_addr as CUdeviceptr,
            ordinal,
            vec![1],
            vec![1],
        )
    });
    let w = unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32, packed_k as i32],
            vec![(n * packed_k) as i32, packed_k as i32, 1],
        )
    };
    let ws = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32, (k / BLOCK_SIZE) as i32],
            vec![(n * k / BLOCK_SIZE) as i32, (k / BLOCK_SIZE) as i32, 1],
        )
    };
    let wg = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            wg_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32],
            vec![n as i32, 1],
        )
    };
    let ag = Arc::new(unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ag_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32],
            vec![1],
        )
    });
    let blocks = (BLOCKS_PER_SM * device_multiprocessor_count(dev)) as u32;
    let a4 = args.activation_global_scale.is_some();
    let generic = vec![
        ROUTED_ROWS.to_string(),
        ROUTED_COLUMNS.to_string(),
        MATMUL_K.to_string(),
        (MATMUL_K / 2).to_string(),
        (MATMUL_K / BLOCK_SIZE).to_string(),
        a4.to_string(),
        LOAD_LATENCY.to_string(),
    ];
    let quant_generic = vec![
        QUANT_ROWS.to_string(),
        QUANT_K.to_string(),
        (QUANT_K / 2).to_string(),
        (QUANT_K / BLOCK_SIZE).to_string(),
        ROUTED_ROWS.to_string(),
        a4.to_string(),
    ];
    macro_rules! dispatch {
        ($launcher:expr) => {{
            let launcher = $launcher;
            if compile_only {
                catch_cutile_panic("NVFP4 grouped compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile NVFP4 grouped compile failed: {error:?}"
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("NVFP4 grouped launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile NVFP4 grouped launch failed: {error:?}"
                        ))
                    })
                })?;
            }
        }};
    }
    macro_rules! run {
        ($dtype:ty, $quant:path, $matmul:path) => {{
            let (x_addr, _x_guard) = slice_ptr_on_stream(
                x_cuda.as_cuda_slice::<$dtype>()?,
                x_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<$dtype>(routes * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let mut packed = unsafe { dev.alloc::<u8>(if a4 { em * packed_k } else { 1 })? };
            let mut scales =
                unsafe { dev.alloc::<F8E4M3>(if a4 { em * k / BLOCK_SIZE } else { 1 })? };
            let mut aligned = unsafe { dev.alloc::<$dtype>(if a4 { 1 } else { em * k })? };
            let (q_addr, q_guard) = slice_ptr_mut_on_stream(&mut packed, 0, &stream);
            let (qs_addr, qs_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
            let (xa_addr, xa_guard) = slice_ptr_mut_on_stream(&mut aligned, 0, &stream);
            let q = unsafe {
                cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
                    (if a4 { q_addr } else { xa_addr }) as CUdeviceptr,
                    ordinal,
                    vec![em as i32, packed_k as i32],
                    vec![packed_k as i32, 1],
                )
            };
            let qs = unsafe {
                cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
                    qs_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![em as i32, (k / BLOCK_SIZE) as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![(k / BLOCK_SIZE) as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            };
            let xa = unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    xa_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![1, 1]
                    } else {
                        vec![em as i32, k as i32]
                    },
                    if a4 { vec![1, 1] } else { vec![k as i32, 1] },
                )
            };
            let q_read = Arc::new(unsafe {
                cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
                    q_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![em as i32, packed_k as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![packed_k as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            });
            let quant = q.partition([QUANT_ROWS, QUANT_K / 2]).map(
                [1, 1],
                blocks.min((em.div_ceil(QUANT_ROWS) * k.div_ceil(QUANT_K)) as u32),
            );
            let quant = unsafe {
                $quant(
                    quant,
                    DevicePointer::<f8e4m3fn>::from_cu_deviceptr(qs_addr as CUdeviceptr),
                    em as i32,
                    (k / BLOCK_SIZE) as i32,
                    DevicePointer::<$dtype>::from_cu_deviceptr(x_addr as CUdeviceptr),
                    DevicePointer::<$dtype>::from_cu_deviceptr(xa_addr as CUdeviceptr),
                    sids.clone(),
                    eids.clone(),
                    ntpp.clone(),
                    routes as i32,
                    x_stride as i32,
                    ag.clone(),
                )
            }
            .generics(quant_generic);
            dispatch!(quant);
            let matmul = unsafe {
                $matmul(
                    DevicePointer::<$dtype>::from_cu_deviceptr(out_addr as CUdeviceptr),
                    sids,
                    eids,
                    ntpp,
                    routes as i32,
                    n as i32,
                    Arc::new(xa),
                    q_read,
                    Arc::new(qs),
                    Arc::new(w),
                    Arc::new(ws),
                    Arc::new(wg),
                    ag,
                )
            }
            .generics(generic)
            .grid((
                (em.div_ceil(ROUTED_ROWS) * n.div_ceil(ROUTED_COLUMNS)) as u32,
                1,
                1,
            ));
            dispatch!(matmul);
            drop(q_guard);
            drop(qs_guard);
            drop(xa_guard);
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[tokens, topk, n]),
            ))
        }};
    }
    match x.dtype() {
        DType::BF16 => Ok(run!(
            bf16,
            kernels::route_quantize_bf16,
            kernels::routed_matmul_bf16
        )),
        DType::F16 => Ok(run!(
            f16,
            kernels::route_quantize_f16,
            kernels::routed_matmul_f16
        )),
        dtype => {
            candle_core::bail!("cuTile NVFP4 grouped matmul does not support {dtype:?} activations")
        }
    }
}
