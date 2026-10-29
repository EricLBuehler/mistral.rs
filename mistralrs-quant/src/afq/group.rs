use std::sync::Arc;

use candle_core::{backend::BackendStorage, MetalStorage, Result, Shape, Storage, Tensor, D};

use crate::{
    ActivationCalibration, GluActivationType, GroupEntry, GroupKey, GroupKind, GroupMember,
};

use super::{AfqInner, AfqLayer};

const AFQ_GROUP_FORMAT: &str = "afq";

impl AfqLayer {
    pub(super) fn afq_group_member(
        &self,
        xs: &Tensor,
        kind: GroupKind<'_>,
    ) -> Result<Option<GroupMember>> {
        if !xs.device().is_metal() || !matches!(kind, GroupKind::Dense | GroupKind::Glu { .. }) {
            return Ok(None);
        }
        let Ok((rows, packed_columns)) = self.w_q.dims2() else {
            return Ok(None);
        };
        Ok(Some(GroupMember {
            key: GroupKey {
                format: AFQ_GROUP_FORMAT,
                dtype: xs.dtype(),
                shape: [1, rows, packed_columns],
                calibration: ActivationCalibration::Dynamic,
            },
            forward: forward_group,
            operand: Arc::new(AfqInner {
                w_q: self.w_q.clone(),
                scales: self.scales.clone(),
                biases: self.biases.clone(),
                bias: self.bias.clone(),
                bits: self.bits,
                group_size: self.group_size,
            }),
        }))
    }
}

fn forward_group(
    xs: &Tensor,
    entries: &[GroupEntry<'_>],
    kind: GroupKind<'_>,
) -> Result<Option<Vec<Tensor>>> {
    if entries.iter().any(|entry| entry.method.has_bias()) {
        return Ok(None);
    }
    let Some(inners) = entries
        .iter()
        .map(|entry| entry.operand::<AfqInner>(AFQ_GROUP_FORMAT))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    match (kind, inners.as_slice()) {
        (GroupKind::Glu { activation }, [gate, up]) => {
            Ok(gate_up(xs, gate, up, activation)?.map(|output| vec![output]))
        }
        (GroupKind::Dense, [q, k, v]) => Ok(qkv(xs, q, k, v)?.map(|(q, k, v)| vec![q, k, v])),
        _ => Ok(None),
    }
}

/// One Metal kernel for both matmuls with shared x reads; the GLU is applied in-register.
fn gate_up(
    xs: &Tensor,
    gi: &AfqInner,
    ui: &AfqInner,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    if gi.bits != ui.bits || gi.group_size != ui.group_size {
        return Ok(None);
    }
    if gi.scales.dtype() != ui.scales.dtype() {
        return Ok(None);
    }
    if gi.w_q.rank() != 2 || ui.w_q.rank() != 2 {
        return Ok(None);
    }
    let k = xs.dim(D::Minus1)?;
    let n_gate = gi.w_q.dim(0)?;
    let n_up = ui.w_q.dim(0)?;
    if n_gate != n_up {
        return Ok(None);
    }
    let n = n_gate;
    // qmm_t uses BM=32 tiles, mostly empty at decode M; separate qmv forwards win there.
    let probe_m = xs.elem_count() / k;
    if probe_m < 16 {
        return Ok(None);
    }
    if k * gi.bits as usize / 8 / 4 != gi.w_q.dim(1)? {
        // unexpected pack factor; let the generic path handle it
        return Ok(None);
    }

    let act_code: u32 = match activation {
        GluActivationType::Silu => 0,
        GluActivationType::Gelu => 1,
        GluActivationType::GeluErf => 2,
        GluActivationType::Relu => 3,
        GluActivationType::Sigmoid => return Ok(None),
    };

    let xs = xs.contiguous()?;
    let m = xs.elem_count() / k;
    if m == 0 {
        return Ok(None);
    }

    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Metal(xs_storage) = &*xs_storage else {
        return Ok(None);
    };
    let (g_w_s, _) = gi.w_q.storage_and_layout();
    let Storage::Metal(g_w_s) = &*g_w_s else {
        return Ok(None);
    };
    let (g_s_s, _) = gi.scales.storage_and_layout();
    let Storage::Metal(g_s_s) = &*g_s_s else {
        return Ok(None);
    };
    let (g_b_s, _) = gi.biases.storage_and_layout();
    let Storage::Metal(g_b_s) = &*g_b_s else {
        return Ok(None);
    };
    let (u_w_s, _) = ui.w_q.storage_and_layout();
    let Storage::Metal(u_w_s) = &*u_w_s else {
        return Ok(None);
    };
    let (u_s_s, _) = ui.scales.storage_and_layout();
    let Storage::Metal(u_s_s) = &*u_s_s else {
        return Ok(None);
    };
    let (u_b_s, _) = ui.biases.storage_and_layout();
    let Storage::Metal(u_b_s) = &*u_b_s else {
        return Ok(None);
    };

    let device = xs_storage.device().clone();
    let dtype = xs.dtype();
    let mut out_shape = xs.dims().to_vec();
    *out_shape.last_mut().unwrap() = n;
    let out = device.new_buffer(out_shape.iter().product(), dtype, "afq-gate-up-out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("afq-gate-up");

    crate::metal_kernels::call_afq_qmm_gate_up(
        device.device(),
        &encoder,
        crate::metal_kernels::Kernels::global(),
        dtype,
        (
            xs_storage.buffer(),
            xs_layout.start_offset() * dtype.size_in_bytes(),
        ),
        g_w_s.buffer(),
        g_s_s.buffer(),
        g_b_s.buffer(),
        u_w_s.buffer(),
        u_s_s.buffer(),
        u_b_s.buffer(),
        &out,
        m,
        n,
        k,
        gi.bits as usize,
        gi.group_size as usize,
        act_code,
    )
    .map_err(candle_core::Error::wrap)?;

    let out_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out,
            device.clone(),
            out_shape.iter().product(),
            dtype,
        )),
        Shape::from(out_shape),
    ));
    Ok(Some(out_t))
}

/// One Metal kernel for all three projections, routing each tile to its weight matrix.
fn qkv(
    xs: &Tensor,
    qi: &AfqInner,
    ki: &AfqInner,
    vi: &AfqInner,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if qi.bits != ki.bits || qi.bits != vi.bits {
        return Ok(None);
    }
    if qi.group_size != ki.group_size || qi.group_size != vi.group_size {
        return Ok(None);
    }
    if qi.scales.dtype() != ki.scales.dtype() || qi.scales.dtype() != vi.scales.dtype() {
        return Ok(None);
    }
    if qi.w_q.rank() != 2 || ki.w_q.rank() != 2 || vi.w_q.rank() != 2 {
        return Ok(None);
    }
    let n_q = qi.w_q.dim(0)?;
    let n_k = ki.w_q.dim(0)?;
    let n_v = vi.w_q.dim(0)?;
    // The kernel routes by tile-aligned column boundaries, so N_q and N_k must be multiples of 32.
    if n_q % 32 != 0 || n_k % 32 != 0 || n_v % 32 != 0 {
        return Ok(None);
    }
    let k_dim = xs.dim(D::Minus1)?;
    // qmm_t uses BM=32 tiles, mostly empty at decode M; separate qmv forwards win there.
    let probe_m = xs.elem_count() / k_dim;
    if probe_m < 16 {
        return Ok(None);
    }

    let xs = xs.contiguous()?;
    let m = xs.elem_count() / k_dim;
    if m == 0 {
        return Ok(None);
    }

    let (xs_s, xs_l) = xs.storage_and_layout();
    let Storage::Metal(xs_s) = &*xs_s else {
        return Ok(None);
    };
    let qws = qi.w_q.storage_and_layout().0;
    let qss = qi.scales.storage_and_layout().0;
    let qbs = qi.biases.storage_and_layout().0;
    let kws = ki.w_q.storage_and_layout().0;
    let kss = ki.scales.storage_and_layout().0;
    let kbs = ki.biases.storage_and_layout().0;
    let vws = vi.w_q.storage_and_layout().0;
    let vss = vi.scales.storage_and_layout().0;
    let vbs = vi.biases.storage_and_layout().0;
    let (Storage::Metal(qw_m), Storage::Metal(qs_m), Storage::Metal(qb_m)) = (&*qws, &*qss, &*qbs)
    else {
        return Ok(None);
    };
    let (Storage::Metal(kw_m), Storage::Metal(ks_m), Storage::Metal(kb_m)) = (&*kws, &*kss, &*kbs)
    else {
        return Ok(None);
    };
    let (Storage::Metal(vw_m), Storage::Metal(vs_m), Storage::Metal(vb_m)) = (&*vws, &*vss, &*vbs)
    else {
        return Ok(None);
    };

    let device = xs_s.device().clone();
    let dtype = xs.dtype();
    let mut q_shape = xs.dims().to_vec();
    let mut k_shape = q_shape.clone();
    let mut v_shape = q_shape.clone();
    *q_shape.last_mut().unwrap() = n_q;
    *k_shape.last_mut().unwrap() = n_k;
    *v_shape.last_mut().unwrap() = n_v;
    let q_out = device.new_buffer(q_shape.iter().product(), dtype, "afq-qkv-q")?;
    let k_out = device.new_buffer(k_shape.iter().product(), dtype, "afq-qkv-k")?;
    let v_out = device.new_buffer(v_shape.iter().product(), dtype, "afq-qkv-v")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("afq-qkv");

    crate::metal_kernels::call_afq_qmm_qkv(
        device.device(),
        &encoder,
        crate::metal_kernels::Kernels::global(),
        dtype,
        (xs_s.buffer(), xs_l.start_offset() * dtype.size_in_bytes()),
        qw_m.buffer(),
        qs_m.buffer(),
        qb_m.buffer(),
        kw_m.buffer(),
        ks_m.buffer(),
        kb_m.buffer(),
        vw_m.buffer(),
        vs_m.buffer(),
        vb_m.buffer(),
        &q_out,
        &k_out,
        &v_out,
        m,
        n_q,
        n_k,
        n_v,
        k_dim,
        qi.bits as usize,
        qi.group_size as usize,
    )
    .map_err(candle_core::Error::wrap)?;

    let q_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            q_out,
            device.clone(),
            q_shape.iter().product(),
            dtype,
        )),
        Shape::from(q_shape),
    ));
    let k_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            k_out,
            device.clone(),
            k_shape.iter().product(),
            dtype,
        )),
        Shape::from(k_shape),
    ));
    let v_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            v_out,
            device.clone(),
            v_shape.iter().product(),
            dtype,
        )),
        Shape::from(v_shape),
    ));
    Ok(Some((q_t, k_t, v_t)))
}
