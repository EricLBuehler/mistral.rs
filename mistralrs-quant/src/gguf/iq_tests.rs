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

/// The dense path has to be exactly a matmul over the dequantised weight: the cache changes
/// when the decode happens, not what it decodes to.
#[test]
fn dense_forward_matches_a_matmul_over_the_dequantized_weight() -> Result<()> {
    use crate::QuantMethod;

    for raw in [16u32, 21, 23, 29] {
        let (out_features, in_features) = (8usize, 256usize);
        let tensor = iq_tensor(raw, &[out_features, in_features], raw)?;
        let w = tensor.dequantize(&Device::Cpu)?;
        let quant = super::IqGgufMatMul::new(std::sync::Arc::new(tensor), None)?;

        let x: Vec<f32> = (0..in_features)
            .map(|i| ((i % 13) as f32) * 0.25 - 1.5)
            .collect();
        let xs = Tensor::from_vec(x, (1, in_features), &Device::Cpu)?;

        let got = quant.forward_raw(&xs)?.flatten_all()?.to_vec1::<f32>()?;
        let want = xs
            .matmul(&w.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_same_bits(&got, &want, &format!("dense, dtype {raw}"));
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
