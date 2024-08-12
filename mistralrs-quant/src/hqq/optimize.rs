use candle_core::{DType, Result, Tensor};

use super::{HqqAxis, HqqLayer};

pub(crate) struct OptParams {
    pub(crate) lp_norm: f64,
    pub(crate) beta: f64,
    pub(crate) kappa: f64,
    pub(crate) iters: usize,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            lp_norm: 0.7,
            beta: 1e1,
            kappa: 1.01,
            iters: 20,
        }
    }
}

pub(crate) struct OptResults {
    pub(crate) wq: Tensor,
    pub(crate) scale: Tensor,
    pub(crate) zero: Tensor,
}

fn shrink_lp_op(x: &Tensor, beta: f64, lp_norm: f64) -> Result<Tensor> {
    if lp_norm == 1. {
        x.sign()? * (x.abs()? - 1. / beta)?.relu()?
    } else {
        x.sign()?
            * (x.abs()? - ((1. / beta) * x.abs()?.pow(&Tensor::new(lp_norm - 1., x.device())?)?))?
                .relu()?
    }
}

impl HqqLayer {
    // https://github.com/mobiusml/hqq/blob/306e30d9400629523c8e0af70101d8d7073cb3d5/hqq/core/optimize.py#L194
    pub(crate) fn optimize_weights_proximal_legacy(
        tensor: &Tensor,
        scale: &Tensor,
        zero: Tensor,
        min: f64,
        max: f64,
        axis: HqqAxis,
        opt_params: OptParams,
    ) -> Result<OptResults> {
        let OptParams {
            lp_norm,
            mut beta,
            kappa,
            iters,
        } = opt_params;

        let wf = tensor.to_dtype(DType::F32)?;
        let scale = scale.to_dtype(DType::F32)?;
        let mut zero = zero.to_dtype(DType::F32)?;

        let mut best_error = 1e4;
        for _ in 0..iters {
            let wq = wf
                .broadcast_mul(&scale)?
                .broadcast_add(&zero)?
                .round()?
                .clamp(min, max)?;
            let wr = wq.broadcast_sub(&zero)?.broadcast_div(&scale)?;
            let we = shrink_lp_op(&(&wf - &wr)?, beta, lp_norm)?;

            zero = (wq - (&wf - we)?.broadcast_mul(&scale)?)?.mean_keepdim(axis as usize)?;
            beta *= kappa;

            let current_error = (&wf - wr)?
                .abs()?
                .mean_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            if current_error < best_error {
                best_error = current_error;
            } else {
                break;
            }
        }

        let wq = tensor
            .broadcast_mul(&scale)?
            .broadcast_add(&zero)?
            .round()?
            .clamp(min, max)?;
        Ok(OptResults { wq, scale, zero })
    }
}
