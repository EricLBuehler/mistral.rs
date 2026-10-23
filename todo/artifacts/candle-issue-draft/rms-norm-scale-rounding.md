# The fused CPU RMSNorm rounds the scale to the storage dtype, and the two implementations in `ops.rs` disagree with each other

`candle_nn::ops::rms_norm` dispatches to the fused `RmsNorm` CPU kernel, which computes the
RMS scale in f32 and then **rounds it to the storage dtype before dividing**:

```rust
// candle-nn/src/ops.rs, the fused CPU kernel
let sum2 = src.iter().map(|&v| { let v = v.as_(); v * v }).sum::<f32>();
let m = (sum2 / n as f32 + eps).sqrt();
let m = T::from_f32(m).unwrap_or_else(T::nan);   // <- the scale is rounded to T
for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
    *d = *s / m * *alpha
}
```

Rounding the scale applies one relative error of up to a bf16 ulp - about 0.4% - to **every
element of a row, uniformly**, before the weight is applied.

## The same file contains an implementation that does it correctly

`candle_nn::ops::rms_norm_slow`, in the same file, normalises in f32 and rounds the
normalised **value** instead:

```rust
let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
```

So `candle-nn/src/ops.rs` holds two implementations of `rms_norm` and they do not agree: one
rounds the scale, the other rounds the value. `rms_norm` dispatches to the fused one, so the
fused behaviour is what callers get.

We cannot know whether the fused kernel was changed deliberately at some point, and this is not
a claim about intent. The defensible form is: the two paths in the same file disagree, every
reference implementation agrees with `rms_norm_slow`, and the downstream consequence is a wrong
expert selection. Whether the fused behaviour was an intentional change or an oversight, it is
the non-conforming one.

## What every reference implementation does

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

llama.cpp uses ggml's kernel, so it keeps the scale in f32 and rounds the normalised value
into the destination dtype.

All three keep the scale in f32. The fused CPU kernel is the only one of the four that rounds
it.

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

Expected output on that row:

```
fused   bf16 bit fnv1a64: 0xdc0d9859544439de
f32scale bf16 bit fnv1a64: 0xd09c33dab7748d5e

first elements where the two disagree:
  elem     fused          f32-scale
    50  0x00003dc3    0.0952148   0x00003dc2    0.0947266
    51  0x00003dc8    0.0976562   0x00003dc7    0.0971680
    52  0x00003dcb    0.0991211   0x00003dca    0.0986328
    53  0x00003dcf    0.1010742   0x00003dce    0.1005859
    54  0x00003dd3    0.1030273   0x00003dd2    0.1025391

max relative difference   7.8125e-3
bit-exact elements       1480 / 2560
per-row ratio             mean 1.00231254   stddev 0.00000769
```

Three things to look at, in increasing order of how much they say:

- The differing elements differ by **exactly one bf16 ulp** (0x3dc3 against 0x3dc2, 0x3dc8
  against 0x3dc7, and so on). That is the rounding of the scale showing up directly.
- The ratio between the two results is a **single constant per row**: mean 1.00231254 with a
  standard deviation of 7.7e-06. A single constant would give a standard deviation of zero;
  the residual is the bf16 quantisation of the output itself.
- 1480 of 2560 elements are bit-identical, because the rounded scale happens to land on the
  same bf16 value for those rows.

## Why it is easy to miss, and how far it reaches

A per-row uniform factor of about 0.4% is invisible to a per-element closeness check. It does
not change the shape of the output, it does not change which elements are large, and a
mean-squared error or a correlation coefficient on the activations barely moves.

**This affects every bf16 model that routes through `candle_nn::ops::rms_norm`, not just
one.** It is invisible to PCC- or MSE-style gates for exactly the reason above, and it only
becomes visible where a small uniform perturbation is amplified - and the two places that
happens are a top-k router and an argmax chain.

## The downstream consequence we hit

We are porting Kolibri-1, whose router selects the top 6 of 384 experts by `logits + bias`.
The selection boundary on that model sits on differences of order 1e-4 to 1e-2 nats. A 0.4%
error in the norm output that feeds the router is worth 0.05-0.3 nats in the router logits -
three orders of magnitude more than the selection margin.

The symptom was that the model answered "Sydney" for "The capital of Australia is" where every
other implementation - Hugging Face, llama.cpp, and the vLLM primary - answers "Canberra". The
answer flipped on the arithmetic of the norm, not on any semantic difference:

```
before    " Sydney"    (logit 10.62, margin 1.06)
after     " Canberra"  (logit 10.00, margin 0.38)
```

With the scale kept in f32, the same build answers "Canberra" and its agreement with an
independent llama.cpp implementation goes from no matching steps to 185 of 248.

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
value to the storage dtype, then apply the weight", not "round the scale, then normalise".
`rms_norm_slow` in the same file already expresses that, so the two paths can be made to agree.

## Versions

- candle 0.11.0, rev `66a8cf184a5a519671454066b1b9efd446ec9f5c` (the revision mistral.rs
  pins), and the same code at upstream `12bb5d1`.
- Reproduced on aarch64, `-C target-cpu=native`, CPU only.

## Not checked

The fused `CustomOp2` exposes no `cuda_fwd` in the revisions examined, so CUDA may route to a
different kernel. Whether it does was not verified.
