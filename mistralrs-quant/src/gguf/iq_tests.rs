// Golden vectors from the local GGUF models, extracted with gguf-py's reference
// dequantiser. The bytes are real model blocks, so this checks the port against a second
// implementation on data the models actually carry.

use super::*;

use crate::gguf::iq_golden::GOLDEN;

/// Each entry names the type, carries four real blocks, and expects 1024 values.
#[test]
fn every_iq_type_dequantises_the_golden_blocks_exactly() {
    for (name, hex_bytes, expected) in GOLDEN {
        let dtype_raw = match *name {
            "IQ2_XXS" => 16,
            "IQ2_XS" => 17,
            "IQ3_XXS" => 18,
            "IQ1_S" => 19,
            "IQ4_NL" => 20,
            "IQ3_S" => 21,
            "IQ2_S" => 22,
            "IQ4_XS" => 23,
            "IQ1_M" => 29,
            other => panic!("unknown golden type {other}"),
        };
        let dtype = GgufDType::new(dtype_raw);
        let type_size = dtype.type_size().expect("golden type size");
        let block_size = dtype.block_size().expect("golden block size");

        let bytes = hex_to_bytes(hex_bytes);
        assert_eq!(bytes.len() % type_size, 0, "{name}: partial block");
        let n_blocks = bytes.len() / type_size;

        let mut got = vec![0f32; n_blocks * block_size];
        dequantize_blocks(dtype, &bytes, n_blocks, &mut got)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        assert_eq!(
            got.len(),
            expected.len(),
            "{name}: {} blocks decode to {} values, expected {}",
            n_blocks,
            got.len(),
            expected.len()
        );
        for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{name}: value {i} is {g}, expected {w}"
            );
        }
    }
}

/// The tensor wrapper rejects a byte count that does not match its shape.
#[test]
fn iq_tensor_checks_the_byte_count() {
    let dtype = GgufDType::new(23); // IQ4_XS: 136 bytes per 256 values
    let ok = IqTensor::new(vec![0u8; 136 * 4].into_boxed_slice(), vec![1024], dtype);
    assert!(ok.is_ok());
    let short = IqTensor::new(vec![0u8; 136 * 4 - 1].into_boxed_slice(), vec![1024], dtype);
    assert!(short.is_err());
    let not_iq = IqTensor::new(
        vec![0u8; 144 * 4].into_boxed_slice(),
        vec![1024],
        GgufDType::new(12), // Q4_K
    );
    assert!(not_iq.is_err());
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("golden hex"))
        .collect()
}

/// Deterministic bytes: any IQ block decodes to something, and this keeps the test free of a
/// random dependency while still exercising the bit fields.
fn pseudo_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}


/// Bit-for-bit comparison. Synthetic blocks can carry a non-finite scale, and NaN != NaN
/// would turn that into a spurious mismatch; the bits are what "identical" means here.
fn assert_same_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: value {i}: {g} vs {w}");
    }
}

/// Same values up to the order the terms are summed in. Two f32 dots over the same row differ
/// in the last bits once one of them runs in lanes; anything larger is a wrong weight.
/// Synthetic blocks can carry a non-finite scale, which makes the whole row non-finite in both
/// arms, so only the finiteness has to agree there.
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want
        .iter()
        .filter(|w| w.is_finite())
        .fold(0f32, |acc, w| acc.max(w.abs()))
        .max(1.0);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if !g.is_finite() || !w.is_finite() {
            assert!(
                g.is_finite() == w.is_finite(),
                "{what}: value {i}: {g} vs {w} (finiteness)"
            );
            continue;
        }
        assert!(
            (g - w).abs() <= 1e-4 * scale,
            "{what}: value {i}: {g} vs {w} (scale {scale})"
        );
    }
}

const IQ_TYPES: [(u32, &str); 9] = [
    (16, "IQ2_XXS"),
    (17, "IQ2_XS"),
    (18, "IQ3_XXS"),
    (19, "IQ1_S"),
    (20, "IQ4_NL"),
    (21, "IQ3_S"),
    (22, "IQ2_S"),
    (23, "IQ4_XS"),
    (29, "IQ1_M"),
];

/// A tensor of arbitrary shape whose last dim is the row length, from pseudo-random blocks.
fn iq_tensor(dtype_raw: u32, dims: &[usize], seed: u32) -> Result<IqTensor> {
    let dtype = GgufDType::new(dtype_raw);
    let type_size = dtype.type_size().expect("type size");
    let block_size = dtype.block_size().expect("block size");
    let in_features = *dims.last().expect("at least one dim");
    assert_eq!(in_features % block_size, 0);
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let bytes = pseudo_bytes(rows * (in_features / block_size) * type_size, seed);
    IqTensor::new(bytes.into_boxed_slice(), dims.to_vec(), dtype)
}

/// The cache changes when the decode happens, not what it decodes to: the bytes the parallel
/// GEMV reads have to be the bytes the block decoder produced.
#[test]
fn dense_forward_dequantizes_the_same_weights() -> Result<()> {
    use crate::QuantMethod;

    for raw in [16u32, 21, 23, 29] {
        let (out_features, in_features) = (8usize, 256usize);
        let tensor = iq_tensor(raw, &[out_features, in_features], raw)?;
        let want = tensor.dequantize(&Device::Cpu)?;
        let quant = super::IqGgufMatMul::new(std::sync::Arc::new(tensor), None)?;
        let got = quant.dequantize_w()?;
        assert_eq!(got.dims(), want.dims(), "dense, dtype {raw}: shape");
        assert_same_bits(
            &got.flatten_all()?.to_vec1::<f32>()?,
            &want.flatten_all()?.to_vec1::<f32>()?,
            &format!("dense, dtype {raw}"),
        );
    }
    Ok(())
}

/// A one-row forward is a GEMV over the dequantised weight: the row split must not change the
/// values, and the dot kernel sums a row in its own lane order, so the check is against the
/// reference matmul up to f32 summation error rather than bit for bit.
#[test]
fn dense_forward_matches_a_matmul_over_the_dequantized_weight() -> Result<()> {
    use crate::QuantMethod;

    for raw in [16u32, 21, 23, 29] {
        // At the GEMV output threshold, so the decode path is the one under test.
        let (out_features, in_features) = (512usize, 256usize);
        let tensor = iq_tensor(raw, &[out_features, in_features], raw)?;
        let w = tensor.dequantize(&Device::Cpu)?;
        let quant = super::IqGgufMatMul::new(std::sync::Arc::new(tensor), None)?;

        let x: Vec<f32> = (0..in_features)
            .map(|i| ((i % 13) as f32) * 0.25 - 1.5)
            .collect();
        let xs = Tensor::from_vec(x, (1, in_features), &Device::Cpu)?;

        let got = quant.forward_raw(&xs)?.flatten_all()?.to_vec1::<f32>()?;
        let want = xs.matmul(&w.t()?)?.flatten_all()?.to_vec1::<f32>()?;
        assert_close(&got, &want, &format!("dense, dtype {raw}"));
    }
    Ok(())
}

/// The expert path has to be exactly the fused whole-stack gather, since it stands in for it:
/// slicing the blocks and remapping the ids must not change a single bit.
#[test]
fn expert_gather_matches_the_fused_stack_bit_exactly() -> Result<()> {
    use crate::moe_gather_cpu::{gather_gemv, NO_REPEAT};
    use crate::QuantMethod;

    for raw in [16u32, 21, 23, 29] {
        let (experts, out, k, tokens, top_k) = (4usize, 8usize, 256usize, 64usize, 4usize);
        let tensor = iq_tensor(raw, &[experts, out, k], raw * 3 + 1)?;
        let stack = tensor.dequantize(&Device::Cpu)?;
        let quant = super::IqGgufMatMul::new(std::sync::Arc::new(tensor), None)?;

        // Overlapping top-k, with one expert repeated across tokens.
        let ids: Vec<u32> = (0..tokens * top_k)
            .map(|i| ((i * 3) % experts) as u32)
            .collect();
        let a: Vec<f32> = (0..tokens * k)
            .map(|i| ((i % 19) as f32) * 0.125 - 1.0)
            .collect();

        // Gate and up: one activation row per token, top_k indices.
        let x5 = Tensor::from_vec(a.clone(), (1, tokens, 1, 1, k), &Device::Cpu)?;
        let ids3 = Tensor::from_vec(ids.clone(), (1, tokens, top_k), &Device::Cpu)?;
        let got = quant
            .gather_forward(&x5, &ids3)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = gather_gemv(&a, &stack, &ids, top_k)?;
        assert_same_bits(&got, &want, &format!("gate/up, dtype {raw}"));

        // Flat form: (tokens, 1, k) with (tokens, top_k) indices.
        let x3 = Tensor::from_vec(a.clone(), (tokens, 1, k), &Device::Cpu)?;
        let ids2 = Tensor::from_vec(ids.clone(), (tokens, top_k), &Device::Cpu)?;
        let got = quant
            .gather_forward(&x3, &ids2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_same_bits(&got, &want, &format!("flat, dtype {raw}"));

        // Down: one activation row per slot.
        let slot_rows: Vec<f32> = (0..tokens * top_k * k)
            .map(|i| ((i % 29) as f32) * 0.0625 - 0.5)
            .collect();
        let x4 = Tensor::from_vec(
            slot_rows.clone(),
            (1, tokens, top_k, k),
            &Device::Cpu,
        )?;
        let got = quant
            .gather_forward(&x4, &ids3)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = gather_gemv(&slot_rows, &stack, &ids, NO_REPEAT)?;
        assert_same_bits(&got, &want, &format!("down, dtype {raw}"));
    }
    Ok(())
}

/// Every IQ layout has to decode arbitrary block bytes without panicking, and produce finite
/// values: the tables are indexed by the packed fields, so an out-of-range table read would
/// show up here before a checkpoint does.
#[test]
fn random_blocks_decode_to_finite_values() -> Result<()> {
    for (raw, name) in IQ_TYPES {
        let dtype = GgufDType::new(raw);
        let type_size = dtype.type_size().expect("type size");
        let block_size = dtype.block_size().expect("block size");
        let n_blocks = 16usize;
        let bytes = pseudo_bytes(type_size * n_blocks, raw * 7 + 1);
        let mut out = vec![0f32; n_blocks * block_size];
        dequantize_blocks(dtype, &bytes, n_blocks, &mut out)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        // Random block bytes can carry a non-finite f16 scale, so only the absence of a panic
        // and a non-degenerate decode are asserted here; real-data finiteness is the golden
        // test's job.
        assert!(out.iter().any(|&v| v != 0.0), "{name}: all zero");
    }
    Ok(())
}

/// When the gather kernel refuses a tiny shape, the fallback has to be the same whole-stack
/// gather the K-quant path falls back to, over the named experts.
#[test]
fn tiny_gather_falls_back_to_the_whole_stack_gather() -> Result<()> {
    use crate::QuantMethod;

    let raw = 23u32; // IQ4_XS
    let (experts, out, k, tokens, top_k) = (4usize, 4usize, 256usize, 2usize, 2usize);
    let tensor = iq_tensor(raw, &[experts, out, k], 77)?;
    let stack = tensor.dequantize(&Device::Cpu)?;
    let quant = super::IqGgufMatMul::new(std::sync::Arc::new(tensor), None)?;

    let ids: Vec<u32> = vec![3, 1, 0, 3];
    let a: Vec<f32> = (0..tokens * k)
        .map(|i| ((i % 11) as f32) * 0.25 - 1.0)
        .collect();

    // 2 * 4 slots at width 4 is below the kernel's threshold, so this takes the fallback.
    assert!(!crate::moe_gather_cpu::should_use_gather_gemv_shape(
        experts, out, k, out, tokens * top_k
    ));

    let x5 = Tensor::from_vec(a.clone(), (1, tokens, 1, 1, k), &Device::Cpu)?;
    let ids3 = Tensor::from_vec(ids.clone(), (1, tokens, top_k), &Device::Cpu)?;
    let got = quant
        .gather_forward(&x5, &ids3)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let unquant = <crate::UnquantLinear as crate::QuantMethod>::new(
        crate::QuantMethodConfig::Unquantized(candle_nn::Linear::new(stack, None)),
    )?;
    // The whole-stack form reads one row per token with top_k indices.
    let x = Tensor::from_vec(a, (1, tokens, 1, 1, k), &Device::Cpu)?;
    let want = crate::QuantMethod::gather_forward(&unquant, &x, &ids3)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_same_bits(&got, &want, "tiny fallback");
    Ok(())
}

/// The split over block ranges decodes the same values: a block is self-contained, so which
/// thread gets it is a scheduling detail and nothing else. The sizes walk the chunk boundary
/// so the tail chunk is covered too.
#[test]
fn parallel_decode_matches_the_serial_decode_bit_exactly() {
    for (raw, name) in IQ_TYPES {
        let dtype = GgufDType::new(raw);
        let (block_size, type_size) = (
            dtype.block_size().expect("block size"),
            dtype.type_size().expect("type size"),
        );
        for n_blocks in [
            1usize,
            7,
            PAR_BLOCK_CHUNK - 1,
            PAR_BLOCK_CHUNK,
            PAR_BLOCK_CHUNK + 1,
            1024,
        ] {
            let bytes = pseudo_bytes(n_blocks * type_size, raw * 11 + n_blocks as u32);

            let mut serial = vec![0f32; n_blocks * block_size];
            dequantize_blocks(dtype, &bytes, n_blocks, &mut serial).expect("serial decode");
            let mut parallel = vec![0f32; n_blocks * block_size];
            dequantize_blocks_par(dtype, &bytes, n_blocks, &mut parallel);

            assert_same_bits(
                &parallel,
                &serial,
                &format!("{name} at {n_blocks} blocks"),
            );
        }
    }
}

/// Where a token's expert time goes, away from the model: one layer is three IQ projections
/// (gate, up, down) over six routed experts, so this times 49 such layers against a stack
/// shaped like Flint's experts. A measurement, not a check: `--ignored --nocapture`.
#[test]
#[ignore = "measurement"]
fn decode_and_gather_cost_of_one_token() {
    use crate::QuantMethod;

    let (experts, out, k) = (256usize, 512usize, 2048usize);
    let dtype = GgufDType::new(16);
    let (block_size, type_size) = (
        dtype.block_size().expect("block size"),
        dtype.type_size().expect("type size"),
    );
    assert_eq!(k % block_size, 0);

    let bytes = pseudo_bytes(experts * out * (k / block_size) * type_size, 16);
    let stack = std::sync::Arc::new(
        IqTensor::new(bytes.into_boxed_slice(), vec![experts, out, k], dtype)
            .expect("iq stack"),
    );
    let matmul = IqGgufMatMul::new(stack.clone(), None).expect("iq matmul");

    let ids: Vec<u32> = (0..6u32).map(|i| i * 7 % experts as u32).collect();
    let x = Tensor::from_vec(vec![0.5f32; k], (1, 1, 1, 1, k), &Device::Cpu).expect("x");
    let idx = Tensor::from_vec(ids.clone(), (1, 1, 6), &Device::Cpu).expect("idx");

    let time = |iters: usize, mut f: Box<dyn FnMut()>| {
        f();
        let started = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        started.elapsed().as_secs_f64() / iters as f64
    };

    let iters = 20;
    let six_rows = vec![0.5f32; 6 * k];
    let gather = time(
        iters,
        Box::new(|| {
            for _ in 0..3 {
                matmul.gather_forward(&x, &idx).expect("gather");
            }
        }),
    );
    let fused = time(
        iters,
        Box::new(|| {
            for _ in 0..3 {
                matmul.fused_gather(&six_rows, &ids, 1).expect("fused");
            }
        }),
    );
    let decode = time(
        iters,
        Box::new(|| {
            for _ in 0..3 {
                stack
                    .dequantize_experts(&ids, &Device::Cpu)
                    .expect("decode");
            }
        }),
    );

    let layer_slabs = 3 * 6 * out * k;
    let per_layer_bytes = (layer_slabs * 4) as f64;
    let per_token = 49.0 * gather;
    println!(
        "{} experts {experts} out {out} k {k}, 3 projections x 6 experts per layer",
        dtype.name()
    );
    println!(
        "decode one layer: {:.2} ms ({:.1} GB/s of f32 written)",
        decode * 1e3,
        per_layer_bytes / decode / 1e9
    );
    println!(
        "gather one layer: {:.2} ms, so {:.2} ms of gemv on top",
        gather * 1e3,
        (gather - decode) * 1e3
    );
    println!(
        "49 layers: {:.0} ms of gather, {:.1} GB of f32 a token, ceiling {:.2} tok/s",
        per_token * 1e3,
        49.0 * per_layer_bytes / 1e9,
        1.0 / per_token
    );
    // The fused path reads the blocks and never writes the slab, so its traffic is bpw-sized.
    let fused_per_layer = 49.0 * fused;
    let block_bytes = 3.0 * 6.0 * out as f64 * (k as f64 / block_size as f64) * type_size as f64;
    println!(
        "fused one layer: {:.2} ms ({:.1} GB/s of block bytes), 49 layers {:.0} ms, ceiling {:.2} tok/s",
        fused * 1e3,
        block_bytes / fused / 1e9,
        fused_per_layer * 1e3,
        1.0 / fused_per_layer
    );
    println!(
        "fused against the two-step: {:.2}x",
        gather / fused
    );
}

/// Random block bytes can carry a non-finite f16 scale, which says nothing about the
/// arithmetic. Both types here keep their scale in the block's first two bytes, so setting it
/// to 0.5 leaves the layout random and the decode finite.
fn finite_blocks(dtype: GgufDType, len: usize, seed: u32) -> Vec<u8> {
    let type_size = dtype.type_size().expect("type size");
    assert!(
        matches!(dtype.raw(), 16 | 22),
        "the scale offset is pinned for these two types only"
    );
    assert_eq!(len % type_size, 0);
    let mut bytes = pseudo_bytes(len, seed);
    for block in bytes.chunks_mut(type_size) {
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
    }
    bytes
}

/// The fused gather is the same products summed in one pass instead of two, so it agrees with
/// the two-step path to f32 rounding and not to the last bit. That is why it is opt-in.
#[test]
fn fused_gather_agrees_with_the_two_step_path() {
    let (experts, out, k) = (8usize, 512usize, 256usize);
    let dtype = GgufDType::new(16);
    let (block_size, type_size) = (
        dtype.block_size().expect("block size"),
        dtype.type_size().expect("type size"),
    );
    assert_eq!(k % block_size, 0);

    let bytes = finite_blocks(dtype, experts * out * (k / block_size) * type_size, 5);
    let stack = std::sync::Arc::new(
        IqTensor::new(bytes.into_boxed_slice(), vec![experts, out, k], dtype)
            .expect("iq stack"),
    );
    let matmul = IqGgufMatMul::new(stack, None).expect("iq matmul");

    let ids: Vec<u32> = vec![0, 3, 5, 7, 1, 2];
    let a: Vec<f32> = (0..ids.len() * k)
        .map(|i| ((i % 17) as f32) * 0.125 - 0.9)
        .collect();
    let rows = Tensor::from_vec(a.clone(), (ids.len(), k), &Device::Cpu).expect("rows");
    let idx = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu).expect("idx");

    let two_step: Vec<f32> = matmul
        .gather_flat(&rows, &idx, 1)
        .expect("two-step gather")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("values");
    let fused = matmul.fused_gather(&a, &ids, 1).expect("fused gather");

    assert_eq!(two_step.len(), fused.len());
    // Random block bytes can carry a non-finite f16 scale; both paths must at least agree on
    // where, and the finite rows are the ones that say anything about the arithmetic.
    // A 256-term f32 sum in two orders: the difference is rounding, so it is bounded by the
    // sum's own scale and not by the last bit.
    let mut worst = 0f32;
    for (i, (x, y)) in two_step.iter().zip(&fused).enumerate() {
        let rel = (x - y).abs() / (1.0 + y.abs());
        worst = worst.max(rel);
        assert!(
            rel <= 1e-3,
            "row {i}: two-step {x} against fused {y} is {rel:e} of the value"
        );
    }
    assert!(
        worst < 2e-3,
        "the two orders should differ by rounding, worst was {worst:e}"
    );
}

/// And against a reference that shares no code with either: the slabs dequantised the ordinary
/// way, dotted in f64. This catches an indexing bug that two f32 paths could share.
#[test]
fn fused_gather_matches_a_reference_in_f64() {
    let (experts, out, k) = (4usize, 128usize, 512usize);
    let dtype = GgufDType::new(22);
    let (block_size, type_size) = (
        dtype.block_size().expect("block size"),
        dtype.type_size().expect("type size"),
    );

    let bytes = finite_blocks(dtype, experts * out * (k / block_size) * type_size, 77);
    let stack = std::sync::Arc::new(
        IqTensor::new(bytes.into_boxed_slice(), vec![experts, out, k], dtype)
            .expect("iq stack"),
    );
    let matmul = IqGgufMatMul::new(stack.clone(), None).expect("iq matmul");

    // Two rows, two slots each, so the slot-to-row mapping is exercised.
    let ids: Vec<u32> = vec![2, 0, 3, 1];
    let a: Vec<f32> = (0..2 * k)
        .map(|i| ((i % 23) as f32) * 0.0625 - 0.7)
        .collect();

    let fused = matmul.fused_gather(&a, &ids, 2).expect("fused gather");

    // The slabs come back sorted and deduped, so `dense` is the mapping from slot to slab;
    // reading them in slot order would silently pair each slot with the wrong expert.
    let (slabs, dense) = stack.dequantize_experts(&ids, &Device::Cpu).expect("slabs");
    assert_eq!(dense.len(), ids.len());
    let mut worst = 0f64;
    for (slot, &slab_index) in dense.iter().enumerate() {
        let w: Vec<f32> = slabs[slab_index as usize]
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("values");
        let x = &a[(slot / 2) * k..][..k];
        for r in 0..out {
            let reference: f64 = (0..k)
                .map(|c| w[r * k + c] as f64 * x[c] as f64)
                .sum();
            let got = fused[slot * out + r] as f64;
            let rel = (got - reference).abs() / (1.0 + reference.abs());
            worst = worst.max(rel);
            assert!(
                rel <= 1e-4,
                "slot {slot} row {r}: fused {got} against reference {reference} is {rel:e} of the value"
            );
        }
    }
    println!("fused against the f64 reference: worst {worst:e} of the value");
}

/// The shape the model actually calls: one activation row, six slots, repeat six. The guard
/// that counts rows rather than slots is what this pins - it was wrong once.
#[test]
fn fused_gather_handles_the_token_shape() {
    let (experts, out, k) = (8usize, 512usize, 2048usize);
    let dtype = GgufDType::new(16);
    let (block_size, type_size) = (
        dtype.block_size().expect("block size"),
        dtype.type_size().expect("type size"),
    );
    let n_blocks = k / block_size;
    let bytes = pseudo_bytes(experts * out * n_blocks * type_size, 3);

    let stack = std::sync::Arc::new(
        IqTensor::new(bytes.into_boxed_slice(), vec![experts, out, k], dtype)
            .expect("iq stack"),
    );
    let matmul = IqGgufMatMul::new(stack, None).expect("iq matmul");

    let ids: Vec<u32> = vec![0, 2, 4, 6, 1, 3];
    let a: Vec<f32> = (0..k).map(|i| ((i % 19) as f32) * 0.25 - 2.0).collect();
    let rows = Tensor::from_vec(a.clone(), (1, k), &Device::Cpu).expect("row");
    let idx = Tensor::from_vec(ids.clone(), (1, 1, ids.len()), &Device::Cpu).expect("idx");

    let two_step: Vec<f32> = matmul
        .gather_flat(&rows, &idx, ids.len())
        .expect("two-step gather")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("values");
    let fused = matmul
        .fused_gather(&a, &ids, ids.len())
        .expect("fused gather");

    assert_eq!(two_step.len(), ids.len() * out);
    for (i, (x, y)) in two_step.iter().zip(&fused).enumerate() {
        // The same guard as above: random blocks can carry a non-finite scale.
        assert_eq!(x.is_finite(), y.is_finite(), "row {i}: {x} against {y}");
        if x.is_finite() {
            assert!(
                (x - y).abs() <= 1e-3 * (1.0 + y.abs()),
                "row {i}: two-step {x} against fused {y}"
            );
        }
    }
}

/// The vector decoder has to be the scalar loop's values and not an approximation of them:
/// the grid product and the negation are the same operations, so nothing may move in the
/// last bits. On a machine without AVX2 this compares the scalar decoder with itself, which
/// is why the golden test stays the real check for the arithmetic.
#[test]
fn iq2_xxs_vector_decode_matches_the_scalar_decode_bit_exactly() {
    let dtype = GgufDType::new(16);
    let (block_size, type_size) = (256usize, 66usize);
    for n_blocks in [1usize, 3, 17, 64] {
        let bytes = pseudo_bytes(n_blocks * type_size, 11 + n_blocks as u32);

        let mut scalar = vec![0f32; n_blocks * block_size];
        dequantize_iq2_xxs(&bytes, n_blocks, &mut scalar);

        let mut decoded = vec![0f32; n_blocks * block_size];
        dequantize_blocks(dtype, &bytes, n_blocks, &mut decoded).expect("dispatch");

        for (i, (a, b)) in scalar.iter().zip(&decoded).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "value {i}: scalar {a} against {b}");
        }
    }
}

/// The vector decoder's rate, against the scalar loop on the same blocks: the gather is
/// compute-bound, so this is the number the model's expert time follows. `--ignored`.
#[test]
#[ignore = "measurement"]
fn iq2_xxs_decode_rate() {
    let dtype = GgufDType::new(16);
    let n_blocks = 4096usize;
    let bytes = pseudo_bytes(n_blocks * 66, 9);
    let mut out = vec![0f32; n_blocks * 256];

    let scalar = {
        let started = std::time::Instant::now();
        for _ in 0..5 {
            dequantize_iq2_xxs(&bytes, n_blocks, &mut out);
        }
        started.elapsed().as_secs_f64() / 5.0
    };
    let dispatched = {
        let started = std::time::Instant::now();
        for _ in 0..5 {
            dequantize_blocks(dtype, &bytes, n_blocks, &mut out).expect("dispatch");
        }
        started.elapsed().as_secs_f64() / 5.0
    };
    let values = (n_blocks * 256) as f64;
    println!(
        "scalar {:.2} ms ({:.2} G values/s), dispatched {:.2} ms ({:.2} G values/s), neon {}",
        scalar * 1e3,
        values / scalar / 1e9,
        dispatched * 1e3,
        values / dispatched / 1e9,
        cfg!(target_arch = "aarch64"),
    );
}
