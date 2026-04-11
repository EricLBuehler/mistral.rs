//! TurboQuant compression on MLX arrays.
//!
//! Implements WHT (Walsh-Hadamard Transform) rotation followed by 4-bit PolarQuant
//! scalar quantization using MLX array operations. The entire compress/decompress
//! pipeline runs on the MLX Metal device, benefiting from MLX's lazy evaluation
//! and kernel fusion.
//!
//! The dual-stage architecture:
//!   hot window  → PolarQuant (WHT + 4-bit scalar quantization)
//!   cold window → QJL sketch (random projection + 1-bit) — not yet implemented
//!
//! Both stages are expressible as MLX matrix multiplications + elementwise ops.

use mlx_rs::Array;

/// Build a normalized Hadamard matrix of size `dim x dim` via the Sylvester construction.
///
/// The WHT matrix is its own inverse (up to normalization). We normalize by `1/sqrt(dim)`
/// so that `H @ H = I`.
///
/// # Panics
/// Panics if `dim` is not a power of two.
fn build_wht_matrix(dim: usize) -> Vec<f32> {
    assert!(
        dim.is_power_of_two(),
        "WHT requires power-of-two head_dim, got {dim}"
    );
    let mut h = vec![vec![1.0f32; dim]; dim];
    let mut step = 1usize;
    while step < dim {
        for i in (0..dim).step_by(step * 2) {
            for j in i..i + step {
                for k in 0..dim {
                    let a = h[j][k];
                    let b = h[j + step][k];
                    h[j][k] = a + b;
                    h[j + step][k] = a - b;
                }
            }
        }
        step *= 2;
    }
    let norm = 1.0 / (dim as f32).sqrt();
    h.into_iter().flatten().map(|x| x * norm).collect()
}

/// Gaussian-optimal Lloyd-Max codebook for 4-bit scalar quantization.
/// Returns (centroids[16], boundaries[15]).
fn lloyd_max_4bit_codebook() -> ([f32; 16], [f32; 15]) {
    // Precomputed Gaussian-optimal Lloyd-Max codebook for 4-bit (16 levels).
    // These match the turboquant-rs codebook tables.
    let centroids: [f32; 16] = [
        -2.7475, -2.0898, -1.6389, -1.2739, -0.9576, -0.6680, -0.3966, -0.1311, 0.1311, 0.3966,
        0.6680, 0.9576, 1.2739, 1.6389, 2.0898, 2.7475,
    ];
    let boundaries: [f32; 15] = [
        -2.4187, -1.8644, -1.4564, -1.1158, -0.8128, -0.5323, -0.2638, 0.0, 0.2638, 0.5323,
        0.8128, 1.1158, 1.4564, 1.8644, 2.4187,
    ];
    (centroids, boundaries)
}

/// TurboQuant compressor backed by MLX arrays.
///
/// Implements PolarQuant: WHT rotation + 4-bit scalar quantization.
/// The WHT matrix and codebook are precomputed at initialization and stored
/// as MLX arrays on-device.
pub struct MlxTurboQuantCompressor {
    /// Precomputed WHT matrix: [head_dim, head_dim]
    wht: Array,
    /// Head dimension (must be power of two)
    head_dim: usize,
    /// Lloyd-Max centroids: [16]
    centroids: Array,
    /// Lloyd-Max boundaries: [15]
    boundaries: Array,
}

impl MlxTurboQuantCompressor {
    /// Create a new compressor for the given head dimension.
    ///
    /// # Errors
    /// Returns error if `head_dim` is not a power of two.
    pub fn new(head_dim: usize) -> Result<Self, String> {
        if !head_dim.is_power_of_two() {
            return Err(format!(
                "MlxTurboQuantCompressor requires power-of-two head_dim, got {head_dim}"
            ));
        }

        let wht_data = build_wht_matrix(head_dim);
        let wht = Array::from_slice(&wht_data, &[head_dim as i32, head_dim as i32]);

        let (cent_data, bound_data) = lloyd_max_4bit_codebook();
        let centroids = Array::from_slice(&cent_data, &[16]);
        let boundaries = Array::from_slice(&bound_data, &[15]);

        Ok(Self {
            wht,
            head_dim,
            centroids,
            boundaries,
        })
    }

    /// Compress a KV tensor from [seq, heads, head_dim] f32 to packed uint8.
    ///
    /// Pipeline:
    /// 1. Flatten to [N, head_dim] where N = seq * heads
    /// 2. WHT rotation: rotated = flat @ H^T
    /// 3. Scalar quantize each coordinate to 4-bit index using Lloyd-Max boundaries
    /// 4. Pack pairs of 4-bit indices into uint8: [N, head_dim/2]
    /// 5. Reshape to [seq, heads, head_dim/2]
    ///
    /// Returns packed uint8 array. Calls `eval()` to force computation.
    pub fn compress(&self, kv: &Array) -> Result<Array, String> {
        let shape = kv.shape();
        if shape.len() != 3 {
            return Err(format!("compress: expected 3D tensor, got {}D", shape.len()));
        }
        let seq = shape[0];
        let heads = shape[1];
        let hd = shape[2] as usize;
        if hd != self.head_dim {
            return Err(format!(
                "compress: head_dim mismatch: expected {}, got {hd}",
                self.head_dim
            ));
        }
        let n = seq * heads;

        // 1. Flatten to [N, head_dim]
        let flat = kv
            .reshape(&[n, hd as i32])
            .map_err(|e| format!("reshape: {e}"))?;

        // 2. WHT rotation
        let wht_t = self.wht.t().map_err(|e| format!("transpose: {e}"))?;
        let rotated = flat
            .matmul(&wht_t)
            .map_err(|e| format!("matmul: {e}"))?;

        // 3. Scalar quantize: for each element, count how many boundaries it exceeds
        // This is equivalent to torch.bucketize(right=True)
        // We do this on CPU since mlx-rs may not have all the needed bitwise ops
        rotated
            .eval()
            .map_err(|e| format!("eval rotated: {e}"))?;

        let rotated_data: Vec<f32> = rotated
            .as_slice::<f32>()
            .map_err(|e| format!("as_slice: {e}"))?
            .to_vec();

        let (_, boundaries) = lloyd_max_4bit_codebook();

        // Quantize and pack on CPU (bitwise ops may be incomplete in mlx-rs)
        let packed_dim = hd / 2;
        let mut packed = Vec::with_capacity(n as usize * packed_dim);

        for row in rotated_data.chunks(hd) {
            for pair in row.chunks(2) {
                let idx0 = quantize_scalar(pair[0], &boundaries);
                let idx1 = if pair.len() > 1 {
                    quantize_scalar(pair[1], &boundaries)
                } else {
                    0
                };
                packed.push(idx0 | (idx1 << 4));
            }
        }

        // Build MLX array from packed data
        let out_shape = [seq, heads, (hd / 2) as i32];
        // Store as f32 for MLX compatibility (mlx-rs may not support u8 arrays well)
        // Each f32 holds a packed byte value [0, 255]
        let packed_f32: Vec<f32> = packed.iter().map(|&b| b as f32).collect();
        let result = Array::from_slice(&packed_f32, &out_shape);

        result.eval().map_err(|e| format!("eval packed: {e}"))?;
        Ok(result)
    }

    /// Decompress packed uint8 back to [seq, heads, head_dim] f32 tensor.
    ///
    /// Pipeline:
    /// 1. Unpack 4-bit indices from packed f32 values
    /// 2. Dequantize: look up centroids by index
    /// 3. Inverse WHT rotation: restored = dequant @ H
    /// 4. Reshape to [seq, heads, head_dim]
    pub fn decompress(&self, packed: &Array) -> Result<Array, String> {
        let shape = packed.shape();
        if shape.len() != 3 {
            return Err(format!(
                "decompress: expected 3D tensor, got {}D",
                shape.len()
            ));
        }
        let seq = shape[0];
        let heads = shape[1];
        let packed_dim = shape[2] as usize;
        let hd = packed_dim * 2;
        let n = seq * heads;

        if hd != self.head_dim {
            return Err(format!(
                "decompress: head_dim mismatch: expected {}, got {hd}",
                self.head_dim
            ));
        }

        // Read packed data
        packed.eval().map_err(|e| format!("eval packed: {e}"))?;
        let packed_data: Vec<f32> = packed
            .as_slice::<f32>()
            .map_err(|e| format!("as_slice: {e}"))?
            .to_vec();

        let (centroids, _) = lloyd_max_4bit_codebook();

        // 1. Unpack and dequantize on CPU
        let mut dequant_data = Vec::with_capacity(n as usize * hd);
        for &pval in &packed_data {
            let byte = pval as u8;
            let idx0 = (byte & 0x0F) as usize;
            let idx1 = ((byte >> 4) & 0x0F) as usize;
            dequant_data.push(centroids[idx0]);
            dequant_data.push(centroids[idx1]);
        }

        // 2. Build MLX array [N, head_dim]
        let dequant = Array::from_slice(&dequant_data, &[n, hd as i32]);

        // 3. Inverse WHT (H is its own inverse when normalized)
        let restored = dequant
            .matmul(&self.wht)
            .map_err(|e| format!("inv matmul: {e}"))?;

        // 4. Reshape to [seq, heads, head_dim]
        let out = restored
            .reshape(&[seq, heads, hd as i32])
            .map_err(|e| format!("reshape: {e}"))?;

        out.eval().map_err(|e| format!("eval out: {e}"))?;
        Ok(out)
    }
}

/// Quantize a single scalar value to a 4-bit index using Lloyd-Max boundaries.
/// Equivalent to `torch.bucketize(right=True)`.
fn quantize_scalar(x: f32, boundaries: &[f32; 15]) -> u8 {
    // Binary search: find first boundary >= x
    match boundaries.binary_search_by(|b| b.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => (i + 1) as u8, // right=True: insert after equal elements
        Err(i) => i as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht_matrix_orthogonal() {
        let dim = 4;
        let h = build_wht_matrix(dim);
        // H @ H^T should be identity (since H is symmetric and normalized)
        for i in 0..dim {
            for j in 0..dim {
                let dot: f32 = (0..dim).map(|k| h[i * dim + k] * h[j * dim + k]).sum();
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-5, "diagonal [{i},{j}] = {dot}");
                } else {
                    assert!(dot.abs() < 1e-5, "off-diagonal [{i},{j}] = {dot}");
                }
            }
        }
    }

    #[test]
    fn test_compress_output_shape() {
        let comp = MlxTurboQuantCompressor::new(4).unwrap();
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let input = Array::from_slice(&data, &[2, 3, 4]); // [seq=2, heads=3, hd=4]
        let packed = comp.compress(&input).unwrap();
        assert_eq!(packed.shape(), &[2, 3, 2]); // [seq=2, heads=3, hd/2=2]
    }

    #[test]
    fn test_decompress_restores_shape() {
        let comp = MlxTurboQuantCompressor::new(4).unwrap();
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let input = Array::from_slice(&data, &[2, 3, 4]);
        let packed = comp.compress(&input).unwrap();
        let restored = comp.decompress(&packed).unwrap();
        assert_eq!(restored.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_roundtrip_quality() {
        // Generate data from approximate N(0,1) distribution
        let dim = 128;
        let seq = 4;
        let heads = 8;
        let n = seq * heads * dim;

        // Simple pseudo-random N(0,1) via Box-Muller (deterministic for test)
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            // Simple hash-based pseudo-random
            let x = ((i * 2654435761) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
            let y = (((i + 1) * 2654435761) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
            let z = (-2.0 * x.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * y).cos();
            data.push(z);
        }

        let comp = MlxTurboQuantCompressor::new(dim).unwrap();
        let input = Array::from_slice(&data, &[seq as i32, heads as i32, dim as i32]);
        let packed = comp.compress(&input).unwrap();
        let restored = comp.decompress(&packed).unwrap();

        restored.eval().unwrap();
        let restored_data: Vec<f32> = restored.as_slice::<f32>().unwrap().to_vec();

        // Compute cosine similarity
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (a, b) in data.iter().zip(restored_data.iter()) {
            dot += (*a as f64) * (*b as f64);
            norm_a += (*a as f64) * (*a as f64);
            norm_b += (*b as f64) * (*b as f64);
        }
        let cosine_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
        assert!(
            cosine_sim > 0.90,
            "cosine similarity {cosine_sim} below threshold 0.90"
        );
    }

    #[test]
    fn test_non_power_of_two_rejected() {
        assert!(MlxTurboQuantCompressor::new(96).is_err());
        assert!(MlxTurboQuantCompressor::new(100).is_err());
    }

    #[test]
    fn test_quantize_scalar_boundaries() {
        let (_, boundaries) = lloyd_max_4bit_codebook();
        // Value below all boundaries → index 0
        assert_eq!(quantize_scalar(-10.0, &boundaries), 0);
        // Value above all boundaries → index 15
        assert_eq!(quantize_scalar(10.0, &boundaries), 15);
        // Value near zero → should be around index 7 or 8
        let idx = quantize_scalar(0.01, &boundaries);
        assert!(idx == 8, "expected 8 for 0.01, got {idx}");
    }
}
