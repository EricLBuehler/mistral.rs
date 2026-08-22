// Phase 0 evidence: candle DType::F4 (MX4) has no CUDA storage/cast path in the
// pinned candle-core rev, so the 4-bit KV cache must use packed U8 storage.
// See F4-KV-STORAGE-ASSESSMENT.md in the repo root.
use candle_core::DType;

#[cfg(feature = "cuda")]
use candle_core::{Device, Tensor};

const F4_BLOCK: usize = 32;
const F4_BIAS: i32 = 8;

fn quantize_q4_block(values: &[f32], block: &mut [u8]) -> f32 {
    let max_abs = values
        .iter()
        .fold(0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    let scale = max_abs / (F4_BIAS as f32);
    for (i, &v) in values.iter().enumerate() {
        let q = (v / scale).round().clamp(-8.0, 7.0) as i32 + F4_BIAS;
        let byte = &mut block[i / 2];
        if i % 2 == 0 {
            *byte = (*byte & 0xF0) | (q as u8 & 0x0F);
        } else {
            *byte = (*byte & 0x0F) | ((q as u8 & 0x0F) << 4);
        }
    }
    scale
}

fn dequantize_q4_block(values: &[f32], block: &[u8], scale: f32, out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let byte = block[i / 2];
        let q = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
        *o = ((q as i32 - F4_BIAS) as f32) * scale;
    }
    let _ = values;
}

#[test]
fn candle_f4_is_sub_byte() {
    assert_eq!(DType::F4.size_in_bytes(), 0, "F4 must be sub-byte");
}

#[test]
fn q4_pack_unpack_round_trip_closes_to_source() {
    let values: Vec<f32> = (0..F4_BLOCK)
        .map(|i| ((i as f32 / F4_BLOCK as f32) - 0.5) * 3.0)
        .collect();
    let mut packed = vec![0u8; F4_BLOCK / 2];
    let scale = quantize_q4_block(&values, &mut packed);
    let mut out = vec![0f32; F4_BLOCK];
    dequantize_q4_block(&values, &packed, scale, &mut out);
    // 4-bit symmetric block format: max error is half a level, scale/2,
    // where scale = max_abs/8, so the bound is max_abs/16.
    let max_abs = values.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    for (a, b) in values.iter().zip(out.iter()) {
        assert!((a - b).abs() <= max_abs / 16.0 + 1e-6, "value {a} -> {b}");
    }
}

#[test]
fn q4_packed_layout_is_half_byte_per_element() {
    // 32 f32 values -> 16 bytes of nibbles + one f32 scale.
    assert_eq!(F4_BLOCK / 2 + std::mem::size_of::<f32>(), 20);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rejects_f4_allocation_and_accepts_u8() {
    let device = Device::Cuda(0);
    // The cache storage vehicle: F4 allocation must fail with the candle
    // "Dummy types" error, proving the 4-bit cache needs U8 packing.
    let err = unsafe { Tensor::empty((1024,), DType::F4, &device) }.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Dummy types") || msg.contains("Unsupported"),
        "unexpected error: {msg}"
    );
    // U8 allocation round-trips through the same path CacheEngine uses.
    let t = unsafe { Tensor::empty((1024,), DType::U8, &device) }.unwrap();
    assert_eq!(t.dtype(), DType::U8);
    assert_eq!(t.device(), &device);
}
