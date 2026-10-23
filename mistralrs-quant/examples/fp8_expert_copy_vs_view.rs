//! Is the per-expert fp8 dequant's cost the copy of the expert, or the call?
//!
//! The scoped gather dequantises one expert at a time, and the measured cost was about 4 ms
//! per expert against about 0.46 ms of arithmetic. That gap is either the copy of the
//! expert's bytes out of the stack or the fixed cost of the op dispatch.
//!
//! `Tensor::i` copies, and on a one-byte dtype that copy may be per-element. Narrowing the
//! stack's expert axis and squeezing it gives a contiguous view of the same bytes, so the
//! comparison below is copy-against-view with everything else held equal.
//!
//! The two are timed back to back and interleaved, and the ratio is what matters: it is
//! robust to a busy box in a way that an absolute rate is not.
//!
//! Usage: fp8_expert_copy_vs_view <checkpoint-dir> [layer]

use std::time::Instant;

use candle_core::{safetensors, DType, Device, IndexOp, Tensor};
use mistralrs_quant::fp8_blockwise_dequantize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/mnt/models/Aleph-Alpha/Kolibri-1".to_string());
    let layer: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let block = config["quantization_config"]["weight_block_size"]
        .as_u64()
        .unwrap_or(128) as usize;

    let name = format!("model.layers.{layer}.mlp.experts.0.gate_proj.weight");
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{dir}/model.safetensors.index.json"
    ))?)?;
    let shard = index["weight_map"][&name]
        .as_str()
        .ok_or("weight not in the index")?;
    let tensors = safetensors::load(format!("{dir}/{shard}"), &Device::Cpu)?;
    let tensor = tensors.get(&name).ok_or("weight not in the shard")?;
    let (shape, dtype) = (tensor.dims().to_vec(), tensor.dtype());
    println!("weight {name}: {shape:?} {dtype:?} block {block}");

    // A stack of a few copies of it, so the expert axis exists to narrow.
    let experts = 6usize;
    let stack = Tensor::stack(&vec![tensor.clone(); experts], 0)?;
    let scales = Tensor::ones(
        (experts, shape[0].div_ceil(block), shape[1].div_ceil(block)),
        DType::F32,
        &Device::Cpu,
    )?;

    let reps = 20usize;
    let mut copy_ms = 0.0f64;
    let mut view_ms = 0.0f64;
    for round in 0..reps {
        // Copy: Tensor::i copies the expert out of the stack.
        let t0 = Instant::now();
        let copied = stack.i(0)?;
        let deq =
            fp8_blockwise_dequantize(&copied, &scales.i(0)?, vec![block, block], DType::BF16)?;
        copy_ms += t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&deq);

        // View: narrow the expert axis and squeeze it, no copy.
        let t0 = Instant::now();
        let viewed = stack.narrow(0, 0, 1)?.squeeze(0)?;
        let deq = fp8_blockwise_dequantize(
            &viewed,
            &scales.narrow(0, 0, 1)?.squeeze(0)?,
            vec![block, block],
            DType::BF16,
        )?;
        view_ms += t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&deq);

        if round == 0 {
            let a = deq.flatten_all()?.to_vec1::<half::bf16>()?;
            println!("  first value {a:?}");
        }
    }
    println!("copy path: {:.3} ms per dequant", copy_ms / reps as f64);
    println!("view path: {:.3} ms per dequant", view_ms / reps as f64);
    println!("ratio:    {:.2}x", copy_ms / view_ms.max(1e-9));
    Ok(())
}
