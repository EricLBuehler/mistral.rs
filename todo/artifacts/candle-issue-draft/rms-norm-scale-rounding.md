# The fused CPU RMSNorm rounds the scale to the storage dtype, which differs from every reference implementation

`candle_nn::ops::rms_norm` on CPU goes through the fused `RmsNorm` kernel, which computes the
RMS scale in f32 and then rounds it to the storage dtype before dividing:

```rust
// candle-nn/src/ops.rs, the fused CPU kernel
let sum2 = src.iter().map(|&v| { let v = v.as_(); v * v }).sum::<f32>();
let m = (sum2 / n as f32 + eps).sqrt();
let m = T::from_f32(m).unwrap_or_else(T::nan);   // <- scale rounded to T
for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
    *d = *s / m * *alpha
}
```

Rounding the scale applies one relative error of up to a bf16 ulp - about 0.4% - to **every
element of a row, uniformly**, before the weight is applied.

## What every other implementation does

Hugging Face (`LlamaRMSNorm.forward`):

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

ggml (`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_rms_norm`):

```c
const float mean  = sum/ne00;
const float scale = 1.0f/sqrtf(mean + eps);
```

llama.cpp uses ggml's kernel, so it rounds the normalised **value** into the destination dtype
and keeps the scale in f32.

All three keep the scale in f32 and round the normalised value. Candle's fused kernel is the
only one of the four that rounds the scale.

## The same file already contains an implementation that agrees with them

`candle_nn::ops::rms_norm_slow` in the same file does it the reference way:

```rust
let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
```

So the two implementations in `candle-nn/src/ops.rs` disagree with each other, and the fused
one is the outlier against HF, ggml and llama.cpp. Since `rms_norm` dispatches to the fused
path, that is the path users get.

## Reproducing

```rust
use candle_core::{DType, Device, Tensor};
let n = 2560usize;
let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0013).sin()).collect();
let g: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0001).collect();
let xb = Tensor::from_vec(x, n, &Device::Cpu)?.to_dtype(DType::BF16)?;
let gb = Tensor::from_vec(g, n, &Device::Cpu)?.to_dtype(DType::BF16)?;

// the fused path
let fused = candle_nn::ops::rms_norm(&xb, &gb, 1e-6)?;

// the same computation with the scale kept in f32
let xf = xb.to_dtype(DType::F32)?;
let sum2 = xf.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
let mean = (sum2 / n as f64)?;
let inv = (mean + 1e-6f64).sqrt()?.recip()?.broadcast_as(xf.shape())?;
let f32 = xf.broadcast_mul(&inv)?.to_dtype(DType::BF16)?.broadcast_mul(&gb)?;
```

On this row:

```
max relative difference   7.8125e-3
bit-exact elements       1480 / 2560
```

The differences are not scattered noise: the ratio between the two results is a single factor
per row, because the whole row is divided by the same rounded scale.

## Why it is easy to miss

A per-row uniform factor of 0.4% is invisible to a per-element closeness check. It does not
change the shape of the output, it does not change which elements are large, and a mean-squared
error or a correlation coefficient on the activations barely moves. It only shows up where a
small uniform perturbation is amplified - and on a top-k router it is decisive.

### The downstream consequence we hit

We are porting Kolibri-1, whose router selects the top 6 of 384 experts by `logits + bias`.
The selection boundary on that model sits on differences of order 1e-4 to 1e-2 nats. A 0.4%
error in the norm output that feeds the router is worth 0.05-0.3 nats in the router logits -
three orders of magnitude more than the selection margin.

The symptom was that the model answered "Sydney" for "The capital of Australia is" where every
other implementation - Hugging Face, llama.cpp, and the vLLM primary - answers "Canberra". The
prompt's answer flipped on the arithmetic of the norm, not on any semantic difference: with
the scale kept in f32 the same build answers "Canberra", and the agreement with an independent
llama.cpp implementation goes from no matching steps to 185 of 248.

## Suggested fix

Keep the scale in f32 and round the normalised value, as `rms_norm_slow` and all three
reference implementations do:

```rust
let sum2 = src.iter().map(|&v| { let v = v.as_(); v * v }).sum::<f32>();
let m = (sum2 / n as f32 + eps).sqrt();
for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
    *d = (*s / m) * *alpha
}
```

The subtlety is the rounding point. The reference behaviour is "normalise in f32, round the
value to the storage dtype, then apply the weight", not "round the scale, then normalise". The
existing `rms_norm_slow` in the same file already expresses that, so the two paths can be made
to agree.

## Versions

- candle 0.11.0, rev `66a8cf184a5a519671454066b1b9efd446ec9f5c` (the revision mistral.rs
  pins), and the same code at upstream `12bb5d1`.
- Reproduced on aarch64, `-C target-cpu=native`, CPU only.

## Not checked

The CUDA path was not examined; the fused `CustomOp2` has no `cuda_fwd` in the tree we looked
at, so CUDA may route elsewhere, but that is worth confirming rather than assuming.
