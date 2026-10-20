//! How fast is the CPU blockwise-fp8 dequantisation, per expert?
//!
//! The Kolibri-1 CPU fp8 path costs 74 s per decode token, which is 74 GB of dequantisation
//! at about 1 GB/s. Two things could be wrong: the *scope* (the whole checkpoint is
//! dequantised every forward rather than the selected experts) and the *rate* of the dequant
//! body itself.
//!
//! This measures the rate on one expert slab in isolation, so the two can be separated. The
//! reference implementation dequantises per expert at the gather, writing bf16 straight into
//! the destination rows, at about 11 GB/s of output. If this measures a similar rate, the
//! scope is the whole problem on our side. If it measures about 1 GB/s, the dequant body
//! needs work as well.
//!
//! Usage: fp8_dequant_rate <checkpoint-dir> [layer] [expert]

use std::time::Instant;

use candle_core::{safetensors, DType, Device};
use mistralrs_quant::fp8_blockwise_dequantize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/mnt/models/Aleph-Alpha/Kolibri-1".to_string());
    let layer: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let expert: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let block = config["quantization_config"]["weight_block_size"]
        .as_u64()
        .unwrap_or(128) as usize;

    let name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/model.safetensors.index.json"))?)?;
    let shard = index["weight_map"][&name]
        .as_str()
        .ok_or_else(|| format!("{name} not in the index"))?
        .to_string();

    let dev = Device::Cpu;
    let tensors = safetensors::load(format!("{dir}/{shard}"), &dev)?;
    let w = tensors
        .get(&name)
        .ok_or_else(|| format!("{name} not in the shard"))?
        .clone();
    let s = tensors
        .get(&format!("{name}_scale_inv"))
        .ok_or_else(|| "no scale_inv".to_string())?
        .clone();

    let elems: usize = w.shape().dims().iter().product();
    let out_bytes = elems * 2;
    let in_bytes = elems;
    println!("{name}: {:?} {:?}", w.shape(), w.dtype());
    println!("scale_inv: {:?} {:?}", s.shape(), s.dtype());
    println!("{elems} elements: {in_bytes} bytes in (fp8), {out_bytes} bytes out (bf16)");

    let reps = 5usize;
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let o = fp8_blockwise_dequantize(&w, &s, vec![block, block], DType::BF16)?;
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
        out = Some(o);
    }
    let o = out.expect("no output");
    let rate_out = out_bytes as f64 / best / 1e9;
    let rate_in = in_bytes as f64 / best / 1e9;
    println!("best of {reps}: {best:.4} s");
    println!("  output rate : {rate_out:.2} GB/s of bf16");
    println!("  input  rate : {rate_in:.2} GB/s of fp8");
    println!(
        "  at 2.75 GB of expert traffic per token that is {:.2} s per token",
        2.75 / rate_out
    );
    println!("  output dtype {:?}", o.dtype());
    Ok(())
}
