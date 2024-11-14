use candle_core::{DType, Result, Tensor};

use super::{HqqAxis, HqqLayer, OPTIMIZER_HQQ_DEFAULT_STEPS};

pub(crate) struct OptParams {
    pub(crate) lp_norm: f64,
    pub(crate) beta: f64,
    pub(crate) kappa: f64,
    pub(crate) iters: usize,
}

impl OptParams {
    pub(crate) fn default(optimization_steps: Option<usize>) -> Self {
        Self {
            lp_norm: 0.7,
            beta: 1e1,
            kappa: 1.01,
            iters: optimization_steps.unwrap_or(OPTIMIZER_HQQ_DEFAULT_STEPS),
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
        x.sign()?.broadcast_mul(&(x.abs()? - 1. / beta)?.relu()?)
    } else {
        let pow_exp = Tensor::new(lp_norm - 1., x.device())?
            .broadcast_as(x.shape().clone())?
            .to_dtype(x.dtype())?;
        x.sign()?
            .broadcast_mul(&(x.abs()? - ((1. / beta) * x.abs()?.pow(&pow_exp)?))?.relu()?)
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

        let wf = tensor.clone();
        let scale = scale.to_dtype(wf.dtype())?;
        let mut zero = zero.to_dtype(wf.dtype())?;

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
