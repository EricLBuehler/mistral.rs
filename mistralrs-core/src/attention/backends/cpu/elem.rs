use half::{bf16, f16};

#[cfg(target_arch = "aarch64")]
use super::neon::{dot_f32, mad_f32, scale_f32};

const DOT_CHUNK: usize = 4;

pub(in crate::attention) trait DowncastF32 {
    fn cast(x: f32) -> Self;
}

impl DowncastF32 for f32 {
    #[inline(always)]
    fn cast(x: f32) -> Self {
        x
    }
}

impl DowncastF32 for f16 {
    #[inline(always)]
    fn cast(x: f32) -> Self {
        f16::from_f32(x)
    }
}

impl DowncastF32 for bf16 {
    #[inline(always)]
    fn cast(x: f32) -> Self {
        bf16::from_f32(x)
    }
}

pub(in crate::attention) trait ElemOps: Copy + DowncastF32 {
    const USE_BARRIER_POOL: bool = false;

    fn to_f32(self) -> f32;
    fn dot(a: &[Self], b: &[Self]) -> f32;

    #[inline(always)]
    fn scale_acc(xs: &mut [f32], scale: f32) {
        for v in xs {
            *v *= scale;
        }
    }

    #[inline(always)]
    fn mad(acc: &mut [f32], values: &[Self], scale: f32) {
        for (acc, value) in acc.iter_mut().zip(values.iter()) {
            *acc += value.to_f32() * scale;
        }
    }
}

impl ElemOps for f32 {
    const USE_BARRIER_POOL: bool = true;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> f32 {
        dot_f32(a, b)
    }

    #[inline(always)]
    fn scale_acc(xs: &mut [f32], scale: f32) {
        scale_f32(xs, scale)
    }

    #[inline(always)]
    fn mad(acc: &mut [f32], values: &[Self], scale: f32) {
        mad_f32(acc, values, scale)
    }
}

impl ElemOps for f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> f32 {
        dot_cast(a, b)
    }
}

impl ElemOps for bf16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> f32 {
        dot_cast(a, b)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0f32;
    let chunks = a.len() / DOT_CHUNK;
    for i in 0..chunks {
        let i = i * DOT_CHUNK;
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
    }
    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn scale_f32(xs: &mut [f32], scale: f32) {
    for v in xs {
        *v *= scale;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn mad_f32(acc: &mut [f32], values: &[f32], scale: f32) {
    for (acc, value) in acc.iter_mut().zip(values.iter()) {
        *acc += *value * scale;
    }
}

#[inline(always)]
fn dot_cast<T: ElemOps>(a: &[T], b: &[T]) -> f32 {
    let mut sum = 0f32;
    let chunks = a.len() / DOT_CHUNK;
    for i in 0..chunks {
        let i = i * DOT_CHUNK;
        sum += a[i].to_f32() * b[i].to_f32()
            + a[i + 1].to_f32() * b[i + 1].to_f32()
            + a[i + 2].to_f32() * b[i + 2].to_f32()
            + a[i + 3].to_f32() * b[i + 3].to_f32();
    }
    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i].to_f32() * b[i].to_f32();
    }
    sum
}
