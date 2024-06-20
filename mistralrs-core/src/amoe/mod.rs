use candle_core::{quantized::QMatMul, DType, Device, Result, Tensor, Var, D};
use candle_nn::{linear, Linear, ModuleT, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

pub struct AnyMoeTrainingResult {
    steps: usize,
    loss: f64,
}

pub trait MlpLayer: Send + Sync {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn get_isq_tensors(&mut self) -> Vec<&mut QMatMul>;
}

pub trait TrainableLayer {
    fn get_vars(&self) -> Vec<Var>;
    fn done_training(&mut self);
    fn trainable_params(&self) -> usize;
}

pub trait AnyMoeModel {
    fn inject_moe(&mut self, config: AnyMoeConfig, dtype: DType, device: &Device) -> Result<()>;
}

#[derive(Serialize, Deserialize)]
pub struct AnyMoeConfig {
    hidden_size: usize,
}

pub struct MoeGate {
    lin: Linear,
}

impl ModuleT for MoeGate {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let hidden_states = xs.apply(&self.lin)?;
        if train {
            candle_nn::ops::softmax(&hidden_states, D::Minus1)
        } else {
            candle_nn::ops::softmax_last_dim(&hidden_states)
        }
    }
}

pub struct MoeMlp {
    experts: Vec<Box<dyn MlpLayer>>,
    gate: MoeGate,
    training: bool,
    vars: Vec<Var>,
}

impl MoeMlp {
    /// Create a new MoeMlp layer. By default this is in training mode.
    pub fn new(
        experts: Vec<Box<dyn MlpLayer>>,
        config: AnyMoeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let n_experts = experts.len();
        let var_map = VarMap::new();

        let vb = VarBuilder::from_varmap(&var_map, dtype, dev);
        let vb = vb.pp("moe_gate");

        let vars = var_map.all_vars();
        if vars.is_empty() {
            candle_core::bail!("No vars to train in MoeMlp, perhaps there are no layer?");
        }
        Ok(Self {
            experts,
            gate: MoeGate {
                lin: linear(config.hidden_size, n_experts, vb)?,
            },
            training: true,
            vars,
        })
    }
}

impl TrainableLayer for MoeMlp {
    fn done_training(&mut self) {
        self.training = false;
    }
    fn trainable_params(&self) -> usize {
        self.gate.lin.weight().elem_count() + self.gate.lin.bias().unwrap().elem_count()
    }
    fn get_vars(&self) -> Vec<Var> {
        self.vars.clone()
    }
}

impl MlpLayer for MoeMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // ^ [b, s, h]
        let gate = self.gate.forward_t(xs, self.training)?;
        // ^ [b, s, n_e]
        let gate_expanded = gate.permute((2, 0, 1))?.unsqueeze(0)?;
        // ^ [b, s, n_e] -> [n_e, b, s, 1]
        let mut expert_outputs = Vec::new();
        for expert in &self.experts {
            expert_outputs.push(expert.forward(xs)?);
        }
        let stacked_outputs = Tensor::stack(&expert_outputs, 0)?;
        // ^ [n_e, b, s, h]
        let weighted_outputs = stacked_outputs.broadcast_mul(&gate_expanded)?;
        weighted_outputs.sum(0)
    }

    fn get_isq_tensors(&mut self) -> Vec<&mut QMatMul> {
        if self.training {
            unreachable!("Should not be applying ISQ before training is complete.");
        }

        let mut accum = Vec::new();
        for expert in &mut self.experts {
            accum.extend(expert.get_isq_tensors());
        }
        accum
    }
}
