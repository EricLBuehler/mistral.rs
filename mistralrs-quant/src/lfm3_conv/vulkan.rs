//! Vulkan implementation of the LFM3 gated causal depthwise convolution.
//!
//! One invocation per output element (`[batch, seq, hidden]`) plus one per
//! next-state element (`[batch, hidden, width]`), mirroring the CPU kernel in
//! `lfm3_conv.rs`. The kernel is f32; other dtypes take the CPU round-trip,
//! which is recorded in the Vulkan fallback counter.

use super::{ConvInputs, PARAM_COUNT};
use candle_core::backend::BackendStorage;
use candle_core::vulkan_backend::{glsl, kernels};
use candle_core::{DType, Device, Result, Storage, Tensor};

const KEY: &str = "mistralrs_lfm3_causal_conv1d";

fn source(wg: usize) -> String {
    let mut s = glsl::prelude(wg, 6);
    s += r#"
layout(std430, set = 0, binding = 0) readonly buffer B0 { float x[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { float w[]; };
layout(std430, set = 0, binding = 2) readonly buffer B2 { float st[]; };
layout(std430, set = 0, binding = 3) readonly buffer B3 { float bs[]; };
layout(std430, set = 0, binding = 4) buffer B4 { float o[]; };
layout(std430, set = 0, binding = 5) buffer B5 { float ns[]; };
layout(push_constant) uniform PC {
    int batch; int seq; int hidden; int width;
    int xs0; int xs1; int xs2;
    int ss0; int ss1; int ss2;
    int ws0; int ws2; int bstride;
    int use_state; int has_bias;
    int xoff; int soff; int woff; int boff;
} pc;
layout(local_size_x = WG) in;

float gated(int b, int t, int c) {
    int off = pc.xoff + b * pc.xs0 + t * pc.xs1 + c * pc.xs2;
    return x[off] * x[off + 2 * pc.hidden * pc.xs2];
}

float prev(int b, int c, int k) {
    if (pc.use_state == 0) { return 0.0; }
    return st[pc.soff + b * pc.ss0 + c * pc.ss1 + k * pc.ss2];
}

void main() {
    int i = gid();
    int out_len = pc.batch * pc.seq * pc.hidden;
    int st_len = pc.batch * pc.hidden * pc.width;
    if (i >= out_len + st_len) { return; }
    if (i < out_len) {
        int c = i % pc.hidden;
        int t = (i / pc.hidden) % pc.seq;
        int b = i / (pc.hidden * pc.seq);
        float sum = 0.0;
        for (int tap = 0; tap < pc.width; tap++) {
            int src = t + tap + 1;
            float v = src >= pc.width ? gated(b, src - pc.width, c) : prev(b, c, src);
            sum += v * w[pc.woff + c * pc.ws0 + tap * pc.ws2];
        }
        if (pc.has_bias != 0) { sum += bs[pc.boff + c * pc.bstride]; }
        int gate = pc.xoff + b * pc.xs0 + t * pc.xs1 + (pc.hidden + c) * pc.xs2;
        o[i] = sum * x[gate];
    } else {
        int j = i - out_len;
        int tap = j % pc.width;
        int c = (j / pc.width) % pc.hidden;
        int b = j / (pc.width * pc.hidden);
        int src = pc.seq + tap;
        ns[j] = src >= pc.width ? gated(b, src - pc.width, c) : prev(b, c, src);
    }
}
"#;
    s
}

impl ConvInputs<'_> {
    pub(super) fn vulkan(&self) -> Result<(Tensor, Tensor)> {
        let all_f32 = [
            Some(self.projected),
            Some(self.weight),
            Some(self.state),
            self.bias,
        ]
        .into_iter()
        .flatten()
        .all(|t| t.dtype() == DType::F32);
        if !all_f32 || !kernels::native_enabled() {
            kernels::note_fallback("lfm3_causal_conv1d", None);
            return self.vulkan_round_trip();
        }
        match self.vulkan_native() {
            Ok(out) => Ok(out),
            Err(e) => {
                kernels::note_fallback("lfm3_causal_conv1d", Some(&e));
                self.vulkan_round_trip()
            }
        }
    }

    fn vulkan_native(&self) -> Result<(Tensor, Tensor)> {
        let (x, xl) = self.projected.storage_and_layout();
        let (w, wl) = self.weight.storage_and_layout();
        let (st, sl) = self.state.storage_and_layout();
        let bias = self.bias.map(Tensor::storage_and_layout);
        let (Storage::Vulkan(x), Storage::Vulkan(w), Storage::Vulkan(st)) = (&*x, &*w, &*st) else {
            candle_core::bail!("LFM3 convolution expects Vulkan storage");
        };
        let (bias_buf, boff) = match &bias {
            Some((b, bl)) => match &**b {
                Storage::Vulkan(b) => (b.buf(), bl.start_offset()),
                _ => candle_core::bail!("LFM3 convolution expects Vulkan storage"),
            },
            // Unused when `has_bias == 0`; any valid buffer satisfies the binding.
            None => (w.buf(), 0),
        };
        let p: &[u64; PARAM_COUNT] = &self.params;
        let dev = x.device().clone();
        let out_len = self.output_len();
        let state_len = self.state_len();
        let output = dev.alloc(DType::F32, out_len)?;
        let next_state = dev.alloc(DType::F32, state_len)?;
        let mut push = kernels::Push::default();
        for v in &p[..13] {
            push = push.us(*v as usize)?;
        }
        let push = push
            .i(i32::from(p[13] != 0))
            .i(i32::from(p[14] != 0))
            .us(xl.start_offset())?
            .us(sl.start_offset())?
            .us(wl.start_offset())?
            .us(boff)?;
        let lim = dev.limits();
        dev.launch(
            KEY,
            source,
            &[
                x.buf(),
                w.buf(),
                st.buf(),
                bias_buf,
                output.buf(),
                next_state.buf(),
            ],
            push,
            &[],
            kernels::grid(out_len + state_len, lim.wg, 1, lim.max_groups_x),
        )?;
        Ok(self.tensors(Storage::Vulkan(output), Storage::Vulkan(next_state)))
    }

    /// CPU reference path for dtypes without a Vulkan kernel (fallback already recorded).
    fn vulkan_round_trip(&self) -> Result<(Tensor, Tensor)> {
        let device = self.projected.device().clone();
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu);
        let projected = cpu(self.projected)?;
        let weight = cpu(self.weight)?;
        let state = cpu(self.state)?;
        let bias = self.bias.map(cpu).transpose()?;
        let inputs = ConvInputs {
            projected: &projected,
            weight: &weight,
            bias: bias.as_ref(),
            state: &state,
            params: params_for_cpu(&self.params, &projected, &state, &weight, bias.as_ref()),
            batch: self.batch,
            seq: self.seq,
            hidden: self.hidden,
            width: self.width,
        };
        let (output, next_state) = inputs.cpu()?;
        Ok((output.to_device(&device)?, next_state.to_device(&device)?))
    }
}

/// `to_device` produces contiguous copies, so the strides change.
fn params_for_cpu(
    p: &[u64; PARAM_COUNT],
    projected: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> [u64; PARAM_COUNT] {
    let mut out = *p;
    let (xs, ss, ws) = (projected.stride(), state.stride(), weight.stride());
    out[4] = xs[0] as u64;
    out[5] = xs[1] as u64;
    out[6] = xs[2] as u64;
    out[7] = ss[0] as u64;
    out[8] = ss[1] as u64;
    out[9] = ss[2] as u64;
    out[10] = ws[0] as u64;
    out[11] = ws[2] as u64;
    out[12] = bias.map_or(0, |b| b.stride()[0] as u64);
    out
}
