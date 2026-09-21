//! Int8-activation CPU matmul over packed PTQ1_0 rows: a scalar reference and an AVX2 kernel that agree bit for bit.

use rayon::prelude::*;

use super::ptq1_0::{block_scale, unpack_block_trits, PTQ1_0_BLOCK_BYTES, PTQ1_0_BLOCK_ELEMS};
use super::ptq1_0_pool::{Chunks, Pool};

const INT8_MAX_F: f32 = 127.0;
const LANES: usize = 8;
const VEC_BYTES: usize = 32;
const VECS_PER_BLOCK: usize = PTQ1_0_BLOCK_ELEMS / VEC_BYTES;
pub(super) const ROWS_PER_TASK: usize = 16;
pub(super) const TOKEN_TILE: usize = 32;
pub(super) const K_TILE_BLOCKS: usize = 8;
/// Up to this many tokens the matmul runs on the spin-then-park pool; above it rayon amortizes its wake-up.
pub(super) const POOL_MAX_TOKENS: usize = 4;
const ROW_BLOCK: usize = 4;
pub(super) const TOKEN_BLOCK: usize = 2;

type Elems<T> = [T; PTQ1_0_BLOCK_ELEMS];
type Lane = [i32; LANES];
type Dots<const R: usize, const T: usize> = [[Lane; T]; R];

/// Activations quantized to int8 with one scale (and one sum) per 128 columns, like the CUDA prepare kernel.
struct Acts {
    q: Vec<Elems<i8>>,
    scales: Vec<f32>,
    sums: Vec<i32>,
    blocks_per_row: usize,
}

fn quantize_block(x: &[f32], q: &mut Elems<i8>, scale: &mut f32, sum: &mut i32) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let inv = if amax > 0.0 { INT8_MAX_F / amax } else { 0.0 };
    *scale = amax / INT8_MAX_F;
    let mut total = 0i32;
    for (q, v) in q.iter_mut().zip(x) {
        *q = (v * inv).round_ties_even() as i8;
        total += *q as i32;
    }
    *sum = total;
}

fn quantize(xt: &[f32], in_dim: usize) -> Acts {
    let n_blocks = xt.len() / PTQ1_0_BLOCK_ELEMS;
    let mut acts = Acts {
        q: vec![[0; PTQ1_0_BLOCK_ELEMS]; n_blocks],
        scales: vec![0.0; n_blocks],
        sums: vec![0; n_blocks],
        blocks_per_row: in_dim / PTQ1_0_BLOCK_ELEMS,
    };
    if xt.len() <= POOL_MAX_TOKENS * in_dim {
        for ((x, q), (scale, sum)) in xt
            .chunks(PTQ1_0_BLOCK_ELEMS)
            .zip(acts.q.iter_mut())
            .zip(acts.scales.iter_mut().zip(acts.sums.iter_mut()))
        {
            quantize_block(x, q, scale, sum);
        }
    } else {
        xt.par_chunks(PTQ1_0_BLOCK_ELEMS)
            .zip(acts.q.par_iter_mut())
            .zip(acts.scales.par_iter_mut().zip(acts.sums.par_iter_mut()))
            .for_each(|((x, q), (scale, sum))| quantize_block(x, q, scale, sum));
    }
    acts
}

/// Lane `l` of `dot` sums bytes `4l..4l+4` of each 32-byte vector, the layout `maddubs` then `madd` produce.
///
/// # Safety
/// `decode` and `dot` may use CPU features the caller must have verified.
unsafe trait Int8Kernel {
    unsafe fn decode(block: &[u8; PTQ1_0_BLOCK_BYTES], out: &mut Elems<u8>);
    unsafe fn dot<const R: usize, const T: usize>(
        ts: [&Elems<u8>; R],
        xs: [&Elems<i8>; T],
    ) -> Dots<R, T>;
}

struct Scalar;

unsafe impl Int8Kernel for Scalar {
    #[inline(always)]
    unsafe fn decode(block: &[u8; PTQ1_0_BLOCK_BYTES], out: &mut Elems<u8>) {
        unpack_block_trits(block, out);
    }

    #[inline(always)]
    unsafe fn dot<const R: usize, const T: usize>(
        ts: [&Elems<u8>; R],
        xs: [&Elems<i8>; T],
    ) -> Dots<R, T> {
        std::array::from_fn(|r| {
            std::array::from_fn(|i| {
                std::array::from_fn(|l| {
                    (0..VECS_PER_BLOCK)
                        .flat_map(|v| (0..4).map(move |j| v * VEC_BYTES + l * 4 + j))
                        .map(|at| ts[r][at] as i32 * xs[i][at] as i32)
                        .sum()
                })
            })
        })
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{
        Avx2, Dots, Elems, Int8Kernel, LANES, PTQ1_0_BLOCK_BYTES, VECS_PER_BLOCK, VEC_BYTES,
    };

    const QS_WIDE_END: usize = 16;
    const QS_NARROW_END: usize = 24;
    const QH_START: usize = 24;

    /// Trit `n` of each byte lane, given `pow3 = 3^n << 8` per lane: the low product byte lands in the high half.
    #[inline(always)]
    unsafe fn trits(bytes: __m256i, pow3: __m256i) -> __m256i {
        _mm256_mulhi_epu16(_mm256_mullo_epi16(bytes, pow3), _mm256_set1_epi16(3))
    }

    /// Packs two 16 x u16 vectors into 32 in-order bytes.
    #[inline(always)]
    unsafe fn pack(lo: __m256i, hi: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(lo, hi))
    }

    #[inline(always)]
    unsafe fn store(out: &mut Elems<u8>, vec: usize, v: __m256i) {
        _mm256_storeu_si256(out.as_mut_ptr().add(vec * VEC_BYTES) as *mut __m256i, v);
    }

    unsafe impl Int8Kernel for Avx2 {
        #[inline(always)]
        unsafe fn decode(block: &[u8; PTQ1_0_BLOCK_BYTES], out: &mut Elems<u8>) {
            let p = block.as_ptr();
            let wide = _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i));
            let pow = |n: i16| _mm256_set1_epi16(3i16.pow(n as u32) << 8);
            let w = |n| trits(wide, pow(n));
            store(out, 0, pack(w(0), w(1)));
            store(out, 1, pack(w(2), w(3)));

            let narrow = _mm_loadl_epi64(p.add(QS_WIDE_END) as *const __m128i);
            let narrow16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(narrow, narrow));
            let lanes8 = |lo: i16, hi: i16| {
                _mm256_setr_epi16(
                    lo, lo, lo, lo, lo, lo, lo, lo, hi, hi, hi, hi, hi, hi, hi, hi,
                )
            };
            let n01 = trits(narrow16, lanes8(1 << 8, 3 << 8));
            let n23 = trits(narrow16, lanes8(9 << 8, 27 << 8));
            store(out, 2, pack(w(4), n01));

            let qh = p.add(QH_START);
            let pair = *qh as i32 | (*qh.add(1) as i32) << 8;
            let qh8 = _mm_shuffle_epi8(
                _mm_cvtsi32_si128(pair),
                _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            );
            let tail = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(narrow, qh8));
            let tail_pow = _mm256_setr_epi16(
                81 << 8,
                81 << 8,
                81 << 8,
                81 << 8,
                81 << 8,
                81 << 8,
                81 << 8,
                81 << 8,
                1 << 8,
                1 << 8,
                3 << 8,
                3 << 8,
                9 << 8,
                9 << 8,
                27 << 8,
                27 << 8,
            );
            store(out, 3, pack(n23, trits(tail, tail_pow)));
            debug_assert_eq!(QS_NARROW_END, QH_START);
        }

        #[inline(always)]
        unsafe fn dot<const R: usize, const T: usize>(
            ts: [&Elems<u8>; R],
            xs: [&Elems<i8>; T],
        ) -> Dots<R, T> {
            let mut sums = [[_mm256_setzero_si256(); T]; R];
            for v in 0..VECS_PER_BLOCK {
                let load =
                    |p: *const u8| _mm256_loadu_si256(p.add(v * VEC_BYTES) as *const __m256i);
                let t: [__m256i; R] = std::array::from_fn(|r| load(ts[r].as_ptr()));
                for i in 0..T {
                    let x = load(xs[i].as_ptr() as *const u8);
                    for r in 0..R {
                        sums[r][i] = _mm256_add_epi16(sums[r][i], _mm256_maddubs_epi16(t[r], x));
                    }
                }
            }
            let ones = _mm256_set1_epi16(1);
            std::array::from_fn(|r| {
                std::array::from_fn(|i| {
                    let lane: [i32; LANES] =
                        std::mem::transmute(_mm256_madd_epi16(sums[r][i], ones));
                    lane
                })
            })
        }
    }
}

struct Scratch {
    tile: Vec<Elems<u8>>,
    scales: Vec<f32>,
    acc: Vec<[f32; LANES]>,
    corr: Vec<[f32; K_TILE_BLOCKS]>,
    xscales: Vec<[f32; K_TILE_BLOCKS]>,
    xsums: Vec<[f32; K_TILE_BLOCKS]>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            tile: Vec::new(),
            scales: Vec::new(),
            acc: Vec::new(),
            corr: Vec::new(),
            xscales: Vec::new(),
            xsums: Vec::new(),
        }
    }
}

/// What one micro tile needs to know about the current row/token/k tiles.
struct TileCtx<'a> {
    tile: &'a [Elems<u8>],
    scales: &'a [f32],
    acts: &'a Acts,
    k0: usize,
    kn: usize,
    xscales: &'a [[f32; K_TILE_BLOCKS]],
    xsums: &'a [[f32; K_TILE_BLOCKS]],
    token_base: usize,
    token_width: usize,
}

fn reduce(lanes: &[f32; LANES]) -> f32 {
    let quad: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    (quad[0] + quad[2]) + (quad[1] + quad[3])
}

/// Accumulates a `R x T` block of (row, token) dots over one k tile; `r0`/`t0` are tile-relative.
#[inline(always)]
unsafe fn micro<K: Int8Kernel, const R: usize, const T: usize>(
    cx: &TileCtx,
    acc: &mut [[f32; LANES]],
    corr: &mut [[f32; K_TILE_BLOCKS]],
    (r0, t0): (usize, usize),
) {
    let at = |r: usize, i: usize| (r0 + r) * cx.token_width + t0 + i;
    let mut a: [[[f32; LANES]; T]; R] =
        std::array::from_fn(|r| std::array::from_fn(|i| acc[at(r, i)]));
    let bpr = cx.acts.blocks_per_row;
    let kn = cx.kn;
    let ts: [&[Elems<u8>]; R] = std::array::from_fn(|r| {
        let start = (r0 + r) * K_TILE_BLOCKS;
        &cx.tile[start..start + kn]
    });
    let span = |i: usize| {
        let start = (cx.token_base + t0 + i) * bpr + cx.k0;
        start..start + kn
    };
    let xq: [&[Elems<i8>]; T] = std::array::from_fn(|i| &cx.acts.q[span(i)]);
    let mut scales = [[[0f32; K_TILE_BLOCKS]; T]; R];
    for r in 0..R {
        let start = (r0 + r) * K_TILE_BLOCKS;
        let ws = &cx.scales[start..start + K_TILE_BLOCKS];
        for i in 0..T {
            let (xs, xsum) = (&cx.xscales[t0 + i], &cx.xsums[t0 + i]);
            let c = &mut corr[at(r, i)];
            for j in 0..K_TILE_BLOCKS {
                let scale = ws[j] * xs[j];
                scales[r][i][j] = scale;
                // (t - 1) * x = t * x - x, so the sum term is collected per tile and subtracted from the row total
                c[j] = scale.mul_add(xsum[j], c[j]);
            }
        }
    }
    for j in 0..kn {
        let dots = K::dot::<R, T>(
            std::array::from_fn(|r| &ts[r][j]),
            std::array::from_fn(|i| &xq[i][j]),
        );
        for r in 0..R {
            for i in 0..T {
                for l in 0..LANES {
                    a[r][i][l] = scales[r][i][j].mul_add(dots[r][i][l] as f32, a[r][i][l]);
                }
            }
        }
    }
    for r in 0..R {
        for i in 0..T {
            acc[at(r, i)] = a[r][i];
        }
    }
}

/// Fills `out` (`rows x tokens`, tokens fastest) for a chunk of weight rows.
#[inline(always)]
unsafe fn matmul_rows<K: Int8Kernel>(
    sc: &mut Scratch,
    out: &mut [f32],
    bytes: &[u8],
    acts: &Acts,
    tokens: usize,
) {
    let bpr = acts.blocks_per_row;
    let row_bytes = bpr * PTQ1_0_BLOCK_BYTES;
    let rows = out.len() / tokens;
    sc.tile
        .resize(rows * K_TILE_BLOCKS, [0; PTQ1_0_BLOCK_ELEMS]);
    sc.scales.resize(rows * K_TILE_BLOCKS, 0.0);
    for token_base in (0..tokens).step_by(TOKEN_TILE) {
        let width = TOKEN_TILE.min(tokens - token_base);
        sc.acc.clear();
        sc.acc.resize(rows * width, [0.0; LANES]);
        sc.corr.clear();
        sc.corr.resize(rows * width, [0.0; K_TILE_BLOCKS]);
        for k0 in (0..bpr).step_by(K_TILE_BLOCKS) {
            let kn = K_TILE_BLOCKS.min(bpr - k0);
            for r in 0..rows {
                let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
                for j in 0..kn {
                    let start = (k0 + j) * PTQ1_0_BLOCK_BYTES;
                    let block: &[u8; PTQ1_0_BLOCK_BYTES] =
                        row[start..start + PTQ1_0_BLOCK_BYTES].try_into().unwrap();
                    K::decode(block, &mut sc.tile[r * K_TILE_BLOCKS + j]);
                    sc.scales[r * K_TILE_BLOCKS + j] = block_scale(block);
                }
            }
            sc.xscales.clear();
            sc.xscales.resize(width, [0.0; K_TILE_BLOCKS]);
            sc.xsums.clear();
            sc.xsums.resize(width, [0.0; K_TILE_BLOCKS]);
            for t in 0..width {
                let start = (token_base + t) * bpr + k0;
                for j in 0..kn {
                    sc.xscales[t][j] = acts.scales[start + j];
                    sc.xsums[t][j] = acts.sums[start + j] as f32;
                }
            }
            let cx = TileCtx {
                tile: &sc.tile,
                scales: &sc.scales,
                acts,
                xscales: &sc.xscales,
                xsums: &sc.xsums,
                k0,
                kn,
                token_base,
                token_width: width,
            };
            let mut r = 0;
            while r < rows {
                let rb = if rows - r >= ROW_BLOCK { ROW_BLOCK } else { 1 };
                let mut t = 0;
                while t < width {
                    let tb = if width - t >= TOKEN_BLOCK {
                        TOKEN_BLOCK
                    } else {
                        1
                    };
                    let at = (r, t);
                    match (rb, tb) {
                        (ROW_BLOCK, TOKEN_BLOCK) => {
                            micro::<K, ROW_BLOCK, TOKEN_BLOCK>(&cx, &mut sc.acc, &mut sc.corr, at)
                        }
                        (ROW_BLOCK, _) => {
                            micro::<K, ROW_BLOCK, 1>(&cx, &mut sc.acc, &mut sc.corr, at)
                        }
                        (_, TOKEN_BLOCK) => {
                            micro::<K, 1, TOKEN_BLOCK>(&cx, &mut sc.acc, &mut sc.corr, at)
                        }
                        _ => micro::<K, 1, 1>(&cx, &mut sc.acc, &mut sc.corr, at),
                    }
                    t += tb;
                }
                r += rb;
            }
        }
        for r in 0..rows {
            for t in 0..width {
                out[r * tokens + token_base + t] =
                    reduce(&sc.acc[r * width + t]) - sc.corr[r * width + t].iter().sum::<f32>();
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_rows_avx2(
    sc: &mut Scratch,
    out: &mut [f32],
    bytes: &[u8],
    acts: &Acts,
    tokens: usize,
) {
    matmul_rows::<Avx2>(sc, out, bytes, acts, tokens)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Backend {
    pub(super) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Self::Avx2;
        }
        Self::Scalar
    }
}

/// Returns `[tokens, out_dim]` for already-transformed activations `xt` of shape `[tokens, in_dim]`.
pub(super) fn packed_matmul(
    bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
    xt: &[f32],
    tokens: usize,
) -> Vec<f32> {
    packed_matmul_with(Backend::detect(), bytes, out_dim, in_dim, xt, tokens)
}

pub(super) fn packed_matmul_with(
    backend: Backend,
    bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
    xt: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let row_bytes = in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES;
    let acts = quantize(xt, in_dim);
    let mut by_row = vec![0f32; out_dim * tokens];
    let run_rows = |sc: &mut Scratch, out: &mut [f32], rows: &[u8]| {
        // SAFETY: `Avx2` is only picked by `Backend::detect` after the runtime check
        unsafe {
            match backend {
                Backend::Scalar => matmul_rows::<Scalar>(sc, out, rows, &acts, tokens),
                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => matmul_rows_avx2(sc, out, rows, &acts, tokens),
            }
        }
    };
    if tokens <= POOL_MAX_TOKENS {
        thread_local! {
            static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::new());
        }
        let chunks = Chunks::new(&mut by_row, ROWS_PER_TASK * tokens);
        Pool::global().run(out_dim.div_ceil(ROWS_PER_TASK), &|i| {
            let start = i * ROWS_PER_TASK * row_bytes;
            let rows = &bytes[start..(start + ROWS_PER_TASK * row_bytes).min(bytes.len())];
            // SAFETY: task `i` is the only one touching output chunk `i`
            let out = unsafe { chunks.get(i) };
            SCRATCH.with(|sc| run_rows(&mut sc.borrow_mut(), out, rows));
        });
    } else {
        by_row
            .par_chunks_mut(ROWS_PER_TASK * tokens)
            .zip(bytes.par_chunks(ROWS_PER_TASK * row_bytes))
            .for_each_init(Scratch::new, |sc, (out, rows)| run_rows(sc, out, rows));
    }
    let mut out = vec![0f32; tokens * out_dim];
    for (r, col) in by_row.chunks_exact(tokens).enumerate() {
        for (t, v) in col.iter().enumerate() {
            out[t * out_dim + r] = *v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::ptq1_0::encode_block;

    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    fn random_blocks(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut bytes = Vec::new();
        for _ in 0..n {
            let mut codes = [0u8; PTQ1_0_BLOCK_ELEMS];
            codes.iter_mut().for_each(|c| *c = (lcg(&mut s) % 3) as u8);
            bytes.extend_from_slice(&encode_block(
                &codes,
                0.01 + (lcg(&mut s) % 100) as f32 * 1e-4,
            ));
        }
        bytes
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_decode_matches_reference() {
        if Backend::detect() != Backend::Avx2 {
            return;
        }
        for block in random_blocks(64, 3).as_chunks::<PTQ1_0_BLOCK_BYTES>().0 {
            let mut want = [0u8; PTQ1_0_BLOCK_ELEMS];
            unpack_block_trits(block, &mut want);
            let mut got = [0u8; PTQ1_0_BLOCK_ELEMS];
            // SAFETY: avx2 detected above
            unsafe { decode_avx2(block, &mut got) };
            assert_eq!(got, want);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_avx2(block: &[u8; PTQ1_0_BLOCK_BYTES], out: &mut Elems<u8>) {
        Avx2::decode(block, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_bit_for_bit() {
        if Backend::detect() != Backend::Avx2 {
            return;
        }
        let wide = (K_TILE_BLOCKS + 3) * PTQ1_0_BLOCK_ELEMS;
        for in_dim in [PTQ1_0_BLOCK_ELEMS, wide] {
            for tokens in [1, 2, TOKEN_BLOCK + 1, TOKEN_TILE + TOKEN_BLOCK + 1] {
                let out_dim = ROWS_PER_TASK * 2 + 5;
                let bytes = random_blocks(out_dim * in_dim / PTQ1_0_BLOCK_ELEMS, 11);
                let mut s = 5u64;
                let x: Vec<f32> = (0..tokens * in_dim)
                    .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
                    .collect();
                let a = packed_matmul_with(Backend::Scalar, &bytes, out_dim, in_dim, &x, tokens);
                let b = packed_matmul_with(Backend::Avx2, &bytes, out_dim, in_dim, &x, tokens);
                assert_eq!(a, b, "in_dim {in_dim} tokens {tokens}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "timing"]
    fn machine_ceilings() {
        use std::{arch::x86_64::*, time::Instant};

        const READ_BYTES: usize = 1 << 29;
        const READ_REPS: usize = 6;
        const PEAK_ITERS: usize = 1 << 24;
        const PEAK_CHAINS: usize = 8;
        const MACS_PER_MADDUBS: f64 = 32.0;

        #[target_feature(enable = "avx2")]
        unsafe fn read_chunk(chunk: &[u8]) -> i32 {
            let mut acc = _mm256_setzero_si256();
            for c in chunk.as_chunks::<256>().0 {
                let p = c.as_ptr() as *const __m256i;
                for i in 0..8 {
                    acc = _mm256_xor_si256(acc, _mm256_loadu_si256(p.add(i)));
                }
            }
            _mm256_extract_epi32::<0>(acc)
        }

        #[target_feature(enable = "avx2")]
        unsafe fn peak_chain(seed: i8) -> i32 {
            let a = _mm256_set1_epi8(seed);
            let b = _mm256_set1_epi8(seed.wrapping_add(1));
            let mut acc = [_mm256_setzero_si256(); PEAK_CHAINS];
            for _ in 0..PEAK_ITERS {
                for c in acc.iter_mut() {
                    *c = _mm256_add_epi16(*c, _mm256_maddubs_epi16(std::hint::black_box(a), b));
                }
            }
            let mut sum = _mm256_setzero_si256();
            for c in acc {
                sum = _mm256_add_epi16(sum, c);
            }
            _mm256_extract_epi32::<0>(sum)
        }

        if Backend::detect() != Backend::Avx2 {
            return;
        }
        let threads = rayon::current_num_threads();
        let data = vec![1u8; READ_BYTES];
        let chunk = READ_BYTES / threads;
        let mut best_read = 0f64;
        for _ in 0..READ_REPS {
            let start = Instant::now();
            // SAFETY: avx2 detected above
            let sink: i32 = data
                .par_chunks(chunk)
                .map(|c| unsafe { read_chunk(c) })
                .sum();
            std::hint::black_box(sink);
            best_read = best_read.max(READ_BYTES as f64 / start.elapsed().as_secs_f64() / 1e9);
        }
        let start = Instant::now();
        // SAFETY: avx2 detected above
        let sink: i32 = (0..threads)
            .into_par_iter()
            .map(|t| unsafe { peak_chain(t as i8) })
            .sum();
        std::hint::black_box(sink);
        let macs = threads as f64 * PEAK_ITERS as f64 * PEAK_CHAINS as f64 * MACS_PER_MADDUBS;
        eprintln!("ceiling read GB/s: {best_read:.1}");
        eprintln!(
            "ceiling int8 GMAC/s: {:.0}",
            macs / start.elapsed().as_secs_f64() / 1e9
        );
        eprintln!("ceiling threads: {threads}");
    }
}
