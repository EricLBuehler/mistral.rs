use std::{iter::zip, ops::Mul};

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};

use crate::{
    apply_scalings_to_x, frozenlinear::FrozenLinear, get_maybe_topk_scalings, LinearLayerLike,
    LoraConfig,
};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
    a_adapters: Vec<Linear>,
    b_adapters: Vec<Linear>,
    scale_adapters: Vec<f64>,
    dropout_adapters: Vec<Option<Dropout>>,
    layer_n: usize,
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
        config: &[(String, LoraConfig)],
        vb: &VarBuilder,
        layer_n: usize,
    ) -> Result<Self> {
        let mut a_adapters = Vec::with_capacity(config.len());
        let mut b_adapters = Vec::with_capacity(config.len());
        let mut scale_adapters = Vec::with_capacity(config.len());
        let mut dropout_adapters = Vec::with_capacity(config.len());
        let a_vb = vb.pp("lora_A".to_string());
        let b_vb = vb.pp("lora_B".to_string());
        for (name, cfg) in config.iter() {
            let a_pp = a_vb.pp(name);
            assert!(a_pp.contains_tensor("weight"));
            let a = a_pp.get_with_hints(
                (cfg.rank, linear_config.in_features),
                "weight",
                init::DEFAULT_KAIMING_NORMAL,
            )?;
            let b_pp = b_vb.pp(name);
            assert!(b_pp.contains_tensor("weight"));
            let b =
                b_pp.get_with_hints((linear_config.out_features, cfg.rank), "weight", init::ZERO)?;
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
            layer_n,
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
        let scalings = get_maybe_topk_scalings(scalings, self.layer_n)?;
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
            let mut input_new = input.to_dtype(adapter_a.weight().dtype())?;
            input_new = apply_scalings_to_x(input_new.clone(), &scalings, i)?;

            input_new = if let Some(ref dropout) = adapter_dropout {
                dropout.forward(&input_new, true)?
            } else {
                input_new.clone()
            };

            let res = adapter_b
                .forward(&adapter_a.forward(&input_new)?)?
                .mul(*adapter_scale)?;
            result = (result + res)?;
        }
        Ok(result)
    }
}
