//! Vulkan RoPE: `x [batch, heads, seq, head_dim]` rotated with `cos`/`sin`
//! caches `[rows, rot_dim]` (or `[batch, seq, rot_dim]`), optionally indexed
//! by per-token `positions` (u32, `batch * seq`). One invocation per output
//! element; dimensions past `2 * rot_dim` are copied. The kernel is f32;
//! other dtypes take the CPU round-trip, recorded in the Vulkan fallback
//! counter.

use candle_core::backend::BackendStorage;
use candle_core::op::BackpropOp;
use candle_core::vulkan_backend::{glsl, kernels};
use candle_core::{DType, Device, Layout, Result, Shape, Storage, Tensor, VulkanStorage};

const KEY: &str = "mistralrs_rope_positions";

fn source(wg: usize) -> String {
    let mut s = glsl::prelude(wg, 5);
    s += r#"
layout(std430, set = 0, binding = 0) readonly buffer B0 { float x[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { float cs[]; };
layout(std430, set = 0, binding = 2) readonly buffer B2 { float sn[]; };
layout(std430, set = 0, binding = 3) readonly buffer B3 { uint pos[]; };
layout(std430, set = 0, binding = 4) buffer B4 { float o[]; };
layout(push_constant) uniform PC {
    int n; int heads; int seq; int hd; int rot; int rows; int batch_rows;
    int positioned; int neox; int xoff; int coff; int soff; int poff;
} pc;
layout(local_size_x = WG) in;

void main() {
    int i = gid();
    if (i >= pc.n) { return; }
    int d = i % pc.hd;
    int row = i / pc.hd;
    int base = pc.xoff + row * pc.hd;
    int b = row / (pc.heads * pc.seq);
    int t = row % pc.seq;
    int tok = b * pc.seq + t;
    int crow;
    if (pc.positioned != 0) {
        crow = min(int(pos[pc.poff + tok]), pc.rows - 1);
    } else {
        crow = pc.batch_rows != 0 ? tok : t;
    }
    float v;
    if (d >= 2 * pc.rot) {
        v = x[base + d];
    } else {
        int pair; int xi; int yi; bool first;
        if (pc.neox != 0) {
            pair = d < pc.rot ? d : d - pc.rot;
            xi = pair; yi = pair + pc.rot; first = d < pc.rot;
        } else {
            pair = d / 2;
            xi = 2 * pair; yi = 2 * pair + 1; first = (d % 2) == 0;
        }
        float c = cs[pc.coff + crow * pc.rot + pair];
        float s = sn[pc.soff + crow * pc.rot + pair];
        float a = x[base + xi];
        float bb = x[base + yi];
        v = first ? a * c - bb * s : bb * c + a * s;
    }
    o[i] = v;
}
"#;
    s
}

struct Launch<'a> {
    x: (&'a VulkanStorage, &'a Layout),
    cos: (&'a VulkanStorage, &'a Layout),
    sin: (&'a VulkanStorage, &'a Layout),
    positions: Option<(&'a VulkanStorage, &'a Layout)>,
    is_neox: bool,
}

/// Runs the kernel. Every operand must be contiguous; `cos`/`sin` are
/// `[rows, rot]` when `positions` is given, else `[seq | batch*seq, rot]` or
/// `[batch, seq, rot]`.
fn launch(l: Launch<'_>) -> Result<VulkanStorage> {
    let (x, xl) = l.x;
    let (c, cl) = l.cos;
    let (s, sl) = l.sin;
    let (batch, heads, seq, hd) = xl.shape().dims4()?;
    let rot = *cl.dims().last().unwrap_or(&0);
    let rows = cl.shape().elem_count() / rot.max(1);
    if rot == 0 || rot * 2 > hd || sl.shape() != cl.shape() {
        candle_core::bail!(
            "invalid RoPE shapes {:?} {:?} {:?}",
            xl.shape(),
            cl.shape(),
            sl.shape()
        );
    }
    if l.positions.is_none() && rows != seq && rows != batch * seq {
        candle_core::bail!(
            "RoPE cache rows {rows} are incompatible with batch {batch} and seq {seq}"
        );
    }
    for (layout, what) in [(xl, "input"), (cl, "cos"), (sl, "sin")] {
        if !layout.is_contiguous() {
            candle_core::bail!("RoPE {what} must be contiguous");
        }
    }
    let (pos_buf, poff) = match l.positions {
        Some((p, pl)) => {
            if p.dtype() != DType::U32
                || pl.shape().elem_count() != batch * seq
                || !pl.is_contiguous()
            {
                candle_core::bail!(
                    "RoPE positions must be contiguous u32 with {} entries",
                    batch * seq
                );
            }
            (p.buf(), pl.start_offset())
        }
        None => (c.buf(), 0),
    };
    let n = batch * heads * seq * hd;
    let dev = x.device().clone();
    let out = dev.alloc(DType::F32, n)?;
    let push = kernels::Push::default()
        .us(n)?
        .us(heads)?
        .us(seq)?
        .us(hd)?
        .us(rot)?
        .us(rows)?
        .i(i32::from(rows == batch * seq && rows != seq))
        .i(i32::from(l.positions.is_some()))
        .i(i32::from(l.is_neox))
        .us(xl.start_offset())?
        .us(cl.start_offset())?
        .us(sl.start_offset())?
        .us(poff)?;
    let lim = dev.limits();
    dev.launch(
        KEY,
        source,
        &[x.buf(), c.buf(), s.buf(), pos_buf, out.buf()],
        push,
        &[],
        kernels::grid(n, lim.wg, 1, lim.max_groups_x),
    )?;
    Ok(out)
}

/// Native RoPE for the `RotaryEmb` custom op (no positions). `None` when the
/// case has no kernel (the caller then takes the recorded CPU round-trip).
pub(super) fn custom_op_fwd(
    s1: &VulkanStorage,
    l1: &Layout,
    s2: &VulkanStorage,
    l2: &Layout,
    s3: &VulkanStorage,
    l3: &Layout,
    is_neox: bool,
) -> Option<VulkanStorage> {
    let eligible =
        [s1, s2, s3].iter().all(|s| s.dtype() == DType::F32) && kernels::native_enabled();
    if !eligible {
        kernels::note_fallback("mistralrs-rotary", None);
        return None;
    }
    match launch(Launch {
        x: (s1, l1),
        cos: (s2, l2),
        sin: (s3, l3),
        positions: None,
        is_neox,
    }) {
        Ok(out) => Some(out),
        Err(e) => {
            kernels::note_fallback("mistralrs-rotary", Some(&e));
            None
        }
    }
}

/// RoPE of `q` on a Vulkan device, natively for f32.
pub(super) fn apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let device = q.device().clone();
    let native = q.dtype() == DType::F32
        && cos.dtype() == DType::F32
        && sin.dtype() == DType::F32
        && kernels::native_enabled();
    if native {
        match native_q(q, cos, sin, positions, is_neox, &device) {
            Ok(t) => return Ok(t),
            Err(e) => kernels::note_fallback("rope_positions", Some(&e)),
        }
    } else {
        kernels::note_fallback("rope_positions", None);
    }
    let cpu = Device::Cpu;
    let out = super::cpu_apply_rotary_q(
        &q.to_device(&cpu)?,
        &cos.to_device(&cpu)?,
        &sin.to_device(&cpu)?,
        positions.map(|p| p.to_device(&cpu)).transpose()?.as_ref(),
        is_neox,
    )?;
    out.to_device(&device)
}

fn native_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
    device: &Device,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let cos = cos.to_device(device)?.contiguous()?;
    let sin = sin.to_device(device)?.contiguous()?;
    let positions = positions
        .map(|p| p.to_device(device)?.to_dtype(DType::U32)?.contiguous())
        .transpose()?;
    let (qs, ql) = q.storage_and_layout();
    let (cs, cl) = cos.storage_and_layout();
    let (ss, sl) = sin.storage_and_layout();
    let ps = positions.as_ref().map(Tensor::storage_and_layout);
    let (Storage::Vulkan(qs), Storage::Vulkan(cs), Storage::Vulkan(ss)) = (&*qs, &*cs, &*ss) else {
        candle_core::bail!("Vulkan RoPE expects Vulkan storage");
    };
    let pos = match &ps {
        Some((p, pl)) => match &**p {
            Storage::Vulkan(p) => Some((p, *pl)),
            _ => candle_core::bail!("Vulkan RoPE expects Vulkan storage"),
        },
        None => None,
    };
    let out = launch(Launch {
        x: (qs, ql),
        cos: (cs, cl),
        sin: (ss, sl),
        positions: pos,
        is_neox,
    })?;
    Ok(Tensor::from_storage(
        Storage::Vulkan(out),
        Shape::from(ql.shape().dims()),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    #[test]
    fn vulkan_rope_matches_cpu() -> Result<()> {
        let vk = match Device::new_vulkan(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping vulkan test: {e}");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        let (b, h, s, hd) = (2, 3, 5, 16);
        for rot in [8, 6] {
            let q = Tensor::randn(0f32, 1.0, (b, h, s, hd), &cpu)?;
            let rows = 40;
            let cos = Tensor::randn(0f32, 1.0, (rows, rot), &cpu)?;
            let sin = Tensor::randn(0f32, 1.0, (rows, rot), &cpu)?;
            let positions = Tensor::new(&[3u32, 9, 1, 0, 39, 7, 7, 2, 30, 11], &cpu)?;
            let cos_s = cos.narrow(0, 0, s)?;
            let sin_s = sin.narrow(0, 0, s)?;
            for neox in [true, false] {
                let expected =
                    super::super::cpu_apply_rotary_q(&q, &cos, &sin, Some(&positions), neox)?;
                let actual = super::apply_rotary_q(
                    &q.to_device(&vk)?,
                    &cos.to_device(&vk)?,
                    &sin.to_device(&vk)?,
                    Some(&positions),
                    neox,
                )?;
                let d = (actual.to_device(&cpu)? - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(d < 1e-5, "positioned rot={rot} neox={neox} diff {d}");

                let expected = super::super::cpu_apply_rotary_q(&q, &cos_s, &sin_s, None, neox)?;
                let actual = super::super::apply_rotary(
                    &q.to_device(&vk)?,
                    &cos_s.to_device(&vk)?,
                    &sin_s.to_device(&vk)?,
                    neox,
                )?;
                let d = (actual.to_device(&cpu)? - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(d < 1e-5, "custom op rot={rot} neox={neox} diff {d}");
            }
        }
        let _ = DType::F32;
        let stats = candle_core::vulkan_backend::stats();
        for op in ["rope_positions", "mistralrs-rotary"] {
            assert!(
                !stats.cpu_fallbacks_by_op.iter().any(|(name, _)| name == op),
                "{op} fell back to the CPU: {stats}"
            );
        }
        Ok(())
    }
}
