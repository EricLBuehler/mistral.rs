#[derive(Clone, Copy)]
pub(super) struct MaskInfo<'a> {
    data: &'a [f32],
    dims: &'a [usize],
    b: usize,
    h: usize,
}

impl<'a> MaskInfo<'a> {
    pub(super) fn new(data: &'a [f32], dims: &'a [usize], b: usize, h: usize) -> Self {
        Self { data, dims, b, h }
    }

    #[inline(always)]
    pub(super) fn value(&self, b_i: usize, h_i: usize, q_pos: usize, kv_pos: usize) -> f32 {
        if let Some(idx) = mask_offset(self.dims, self.b, self.h, b_i, h_i, q_pos, kv_pos) {
            *self.data.get(idx).unwrap_or(&0.0)
        } else {
            0.0
        }
    }
}

#[inline(always)]
fn clamp_index(idx: usize, dim: usize) -> usize {
    if dim == 0 {
        0
    } else if idx >= dim {
        dim - 1
    } else {
        idx
    }
}

#[inline]
fn mask_offset(
    dims: &[usize],
    b: usize,
    h: usize,
    b_i: usize,
    h_i: usize,
    q_pos: usize,
    kv_pos: usize,
) -> Option<usize> {
    if dims.is_empty() {
        return Some(0);
    }

    match dims.len() {
        1 => {
            let kv_dim = dims[0];
            if kv_dim == 0 {
                None
            } else {
                Some(clamp_index(kv_pos, kv_dim))
            }
        }
        2 => {
            let q_dim = dims[0];
            let kv_dim = dims[1];
            if q_dim == 0 || kv_dim == 0 {
                None
            } else {
                let q_idx = clamp_index(q_pos, q_dim);
                let kv_idx = clamp_index(kv_pos, kv_dim);
                Some(q_idx * kv_dim + kv_idx)
            }
        }
        3 => {
            let d0 = dims[0];
            let d1 = dims[1];
            let d2 = dims[2];
            if d0 == 0 || d1 == 0 || d2 == 0 {
                return None;
            }
            let q_idx = clamp_index(q_pos, d1);
            let kv_idx = clamp_index(kv_pos, d2);
            if d0 == b || d0 == 1 {
                let b_idx = if d0 == 1 { 0 } else { clamp_index(b_i, d0) };
                Some((b_idx * d1 + q_idx) * d2 + kv_idx)
            } else if d0 == h || d0 == 1 {
                let h_idx = if d0 == 1 { 0 } else { clamp_index(h_i, d0) };
                Some((h_idx * d1 + q_idx) * d2 + kv_idx)
            } else if d0 == b.saturating_mul(h) {
                let combined_idx = clamp_index(b_i * h + h_i, d0);
                Some((combined_idx * d1 + q_idx) * d2 + kv_idx)
            } else {
                Some(q_idx * d2 + kv_idx)
            }
        }
        4 => {
            let d0 = dims[0];
            let d1 = dims[1];
            let d2 = dims[2];
            let d3 = dims[3];
            if d0 == 0 || d1 == 0 || d2 == 0 || d3 == 0 {
                return None;
            }
            let b_idx = if d0 == 1 {
                0
            } else if d0 == b {
                clamp_index(b_i, d0)
            } else if d0 == b.saturating_mul(h) {
                clamp_index(b_i * h + h_i, d0)
            } else {
                clamp_index(b_i, d0)
            };
            let h_idx = if d1 == 1 {
                0
            } else if d1 == h {
                clamp_index(h_i, d1)
            } else if d1 == b {
                clamp_index(b_i, d1)
            } else {
                clamp_index(h_i, d1)
            };
            let q_idx = clamp_index(q_pos, d2);
            let kv_idx = clamp_index(kv_pos, d3);
            Some(((b_idx * d1 + h_idx) * d2 + q_idx) * d3 + kv_idx)
        }
        _ => {
            let q_dim = *dims.get(dims.len().saturating_sub(2))?;
            let kv_dim = *dims.last()?;
            if q_dim == 0 || kv_dim == 0 {
                return None;
            }
            let q_idx = clamp_index(q_pos, q_dim);
            let kv_idx = clamp_index(kv_pos, kv_dim);
            let mut prefix_dim = 1usize;
            for &dim in &dims[..dims.len() - 2] {
                if dim == 0 {
                    return None;
                }
                prefix_dim = prefix_dim.saturating_mul(dim);
            }
            let combined_idx = if prefix_dim == 0 {
                0
            } else {
                let idx_val = b_i * h + h_i;
                idx_val.min(prefix_dim - 1)
            };
            Some((combined_idx * q_dim + q_idx) * kv_dim + kv_idx)
        }
    }
}
