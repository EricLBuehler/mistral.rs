//! Decode-shaped MoE gather benchmark at Kolibri-1's real sizes, with cache effects under
//! control.
//!
//! The first version of this reused six slabs twenty times, which is 15.7 MB and therefore L3
//! resident: it reported 52 GB/s where the model achieves 19. Any optimization judged against
//! that number would have been judged against a cache.
//!
//! So the pool size is a parameter. A pool that fits in cache measures the kernel; a pool that
//! does not measures what decode actually does, where the six selected experts are drawn from
//! 384 and every token lands somewhere cold.
//!
//!   KOLIBRI1_MOE_BENCH=pool=<experts> tokens=<n> threads=<n>
//!   KOLIBRI1_MOE_BENCH=swatcheck   (thread scaling at a fixed pool)

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use mistralrs_quant::moe_gather_cpu;

fn parse_cfg() -> (usize, usize) {
    let arg = std::env::var("KOLIBRI1_MOE_BENCH").unwrap_or_default();
    let get = |k: &str, d: usize| {
        arg.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{k}=")))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    (get("pool", 384), get("tokens", 64))
}

/// One token's routed-expert work: `top_k` slots, each a distinct expert.
///
/// Gate and up read the token's hidden state; down reads the `inter`-wide gate-up product, so
/// they are separate activations, as they are in the model.
fn one_token(gate: &[Tensor], down: &[Tensor], ids: &[u32], xs: &[f32], xs_down: &[f32]) {
    for (pool, a, out_features, in_features) in [
        (gate, xs, 512usize, 2560usize),
        (down, xs_down, 2560usize, 512usize),
    ] {
        let _ = moe_gather_cpu::gather_gemv_per_expert(a, pool, out_features, in_features, ids, 6)
            .expect("gather");
    }
}

#[test]
fn routed_gather_at_scale() {
    let arg = std::env::var("KOLIBRI1_MOE_BENCH").unwrap_or_default();
    if arg.is_empty() {
        return;
    }
    let dev = Device::Cpu;
    let (hidden, inter, top_k) = (2560usize, 512usize, 6usize);
    // Interleave-all for the allocation: this box's node 3 is nearly full, and the point here
    // is cache residency and instruction count, not NUMA placement, which is measured
    // end-to-end instead.
    let (pool_size, tokens) = if arg == "swatcheck" {
        (384usize, 96usize)
    } else {
        parse_cfg()
    };

    // One gate slab and one down slab per expert: the real checkpoint has both, and they are
    // 2.6 MB and 2.6 MB, so 384 experts of each is 2 GB and would not fit. Half the experts
    // gives the same cache pressure per token at a quarter the memory.
    let stride = 2;
    let gate: Vec<Tensor> = (0..pool_size / stride)
        .map(|_| {
            Tensor::rand(-1f32, 1f32, (inter, hidden), &dev)
                .expect("slab")
                .to_dtype(DType::BF16)
                .expect("cast")
        })
        .collect();
    let down: Vec<Tensor> = gate
        .iter()
        .map(|_| {
            Tensor::rand(-1f32, 1f32, (hidden, inter), &dev)
                .expect("slab")
                .to_dtype(DType::BF16)
                .expect("cast")
        })
        .collect();
    let resident = (pool_size / stride) * (inter * hidden + hidden * inter) * 2;
    println!(
        "pool {pool_size} experts ({:.0} MB resident, stride {stride}), {tokens} tokens, \
         threads {}",
        resident as f64 / 1e6,
        mistralrs_quant::cpu_threads::resolve()
    );

    let xs: Vec<f32> = (0..hidden).map(|i| (i % 29) as f32 * 0.1).collect();
    let xs_down: Vec<f32> = (0..inter).map(|i| (i % 13) as f32 * 0.1).collect();

    // A rotation so consecutive tokens select different experts, which is what decode does.
    // Within a token the six are adjacent, so the access is six streams, not six hundred.
    let ids_for = |token: usize| -> Vec<u32> {
        (0..top_k)
            .map(|j| ((token * top_k + j) % (pool_size / stride)) as u32)
            .collect()
    };

    one_token(&gate, &down, &ids_for(0), &xs, &xs_down);
    let t0 = Instant::now();
    for t in 0..tokens {
        one_token(&gate, &down, &ids_for(t), &xs, &xs_down);
    }
    let secs = t0.elapsed().as_secs_f64();
    // Only gate and down are gathered here, at `inter x hidden` and `hidden x inter`. `up` is
    // the same shape as gate and would add a third, but it is not measured.
    let per_token = top_k * 2 * inter * hidden * 2;
    println!(
        "  routed gather: {:.2} GB in {secs:.3}s = {:>6.1} GB/s  ({:.1} ms/token)",
        per_token as f64 * tokens as f64 / 1e9,
        per_token as f64 * tokens as f64 / 1e9 / secs,
        secs * 1e3 / tokens as f64
    );
}
