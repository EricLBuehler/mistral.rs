use std::{collections::HashMap, iter::zip, ops::Mul};

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};

use crate::{apply_scalings_to_x, frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
    a_adapters: Vec<Linear>,
    b_adapters: Vec<Linear>,
    scale_adapters: Vec<f64>,
    dropout_adapters: Vec<Option<Dropout>>,
}

#[derive(Clone, Debug)]
/// Configuration for LoraLinear
pub struct LoraLinearConfig {
    in_features: usize,
    out_features: usize,
}

impl LoraLinearConfig {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LoraLinearConfig {
            in_features,
            out_features,
        }
    }
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &HashMap<String, LoraConfig>,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let mut a_adapters = Vec::with_capacity(config.len());
        let mut b_adapters = Vec::with_capacity(config.len());
        let mut scale_adapters = Vec::with_capacity(config.len());
        let mut dropout_adapters = Vec::with_capacity(config.len());
        let a_vb = vb.pp(format!("lora_A"));
        let b_vb = vb.pp(format!("lora_B"));
        for (_, (name, cfg)) in config.iter().enumerate() {
            let a_pp = a_vb.pp(name);
            assert!(a_pp.contains_tensor("weight"));
            let a = a_pp.get_with_hints(
                (cfg.rank, linear_config.in_features),
                "weight",
                init::DEFAULT_KAIMING_NORMAL,
            )?;
            let b_pp = b_vb.pp(name);
            assert!(a_pp.contains_tensor("weight"));
            let b = b_pp.pp(name).get_with_hints(
                (linear_config.out_features, cfg.rank),
                "weight",
                init::ZERO,
            )?;
            a_adapters.push(Linear::new(a, None));
            b_adapters.push(Linear::new(b, None));
            scale_adapters.push(if cfg.rank > 0 {
                cfg.alpha / cfg.rank as f64
            } else {
                1.0
            });
            dropout_adapters.push(cfg.dropout.map(Dropout::new));
        }

        Ok(LoraLinear {
            old: FrozenLinear::new_from_linear(old)?,
            a_adapters,
            b_adapters,
            scale_adapters,
            dropout_adapters,
        })
    }
}

impl LinearLayerLike for LoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
    fn shape(&self) -> &Shape {
        self.old.shape()
    }
    fn lora_forward(&self, input: &Tensor, scalings: Tensor) -> Result<Tensor> {
        //No fan_in_fan_out so no weight.transpose(0,1)
        let mut result = self.old.forward(input)?;
        for (i, (adapter_a, (adapter_b, (adapter_scale, adapter_dropout)))) in zip(
            &self.a_adapters,
            zip(
                &self.b_adapters,
                zip(&self.scale_adapters, &self.dropout_adapters),
            ),
        )
        .enumerate()
        {
            let input_new = if let Some(ref dropout) = adapter_dropout {
                dropout.forward(input, true)?
            } else {
                input.clone()
            };
            let input_mod = apply_scalings_to_x(input_new.clone(), &scalings, i)?;
            result = (result + adapter_b.forward(&adapter_a.forward(&input_mod)?))?
                .mul(*adapter_scale)?;
        }
        Ok(result)
    }
}
