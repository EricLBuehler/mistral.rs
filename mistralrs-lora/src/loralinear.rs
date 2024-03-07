use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};
use either::Either;
use std::{iter::zip, ops::Mul};

use crate::{
    apply_scalings_to_x, frozenlinear::FrozenLinear, get_maybe_topk_scalings, LinearLayerLike,
    LoraConfig,
};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
    a: Either<Vec<Linear>, Linear>,
    b: Either<Vec<Linear>, Linear>,
    scale_adapters: Vec<f64>,
    dropout: Either<Vec<Option<Dropout>>, Option<Dropout>>,
    layer_n: usize,
    all_same: bool,
    n_adapters: usize,
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
        let mut scale_adapters = Vec::with_capacity(config.len());
        let a_vb = vb.pp("lora_A".to_string());
        let b_vb = vb.pp("lora_B".to_string());

        let rank = config[0].1.rank;
        let dropout = config[0].1.dropout;
        let mut all_same = true;
        for (_, cfg) in config.iter() {
            if cfg.rank != rank || cfg.dropout != dropout {
                all_same = false;
                break;
            }
        }

        let (a, b, dropout) = if !all_same {
            let mut a_adapters = Vec::with_capacity(config.len());
            let mut b_adapters = Vec::with_capacity(config.len());
            let mut dropout_adapters = Vec::with_capacity(config.len());
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
                let b = b_pp.get_with_hints(
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
            (
                Either::Left(a_adapters),
                Either::Left(b_adapters),
                Either::Left(dropout_adapters),
            )
        } else {
            let mut a_adapters = Vec::with_capacity(config.len());
            let mut b_adapters = Vec::with_capacity(config.len());
            let dropout = dropout.map(Dropout::new);
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
                let b = b_pp.get_with_hints(
                    (linear_config.out_features, cfg.rank),
                    "weight",
                    init::ZERO,
                )?;
                a_adapters.push((a * (cfg.alpha / cfg.rank as f64))?);
                b_adapters.push(b);
            }
            let a = Tensor::cat(&a_adapters, 0)?;
            let b = Tensor::cat(&b_adapters, 1)?;
            let a = Linear::new(a, None);
            let b = Linear::new(b, None);
            (Either::Right(a), Either::Right(b), Either::Right(dropout))
        };

        Ok(LoraLinear {
            old: FrozenLinear::new_from_linear(old)?,
            a,
            b,
            scale_adapters,
            dropout,
            layer_n,
            all_same,
            n_adapters: config.len(),
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
    fn lora_forward(
        &self,
        input: &Tensor,
        scalings: Tensor,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let scalings = get_maybe_topk_scalings(scalings, self.layer_n)?;
        //No fan_in_fan_out so no weight.transpose(0,1)
        let mut result = self.old.forward(input)?;

        if !self.all_same {
            for (i, (adapter_a, (adapter_b, (adapter_scale, adapter_dropout)))) in zip(
                self.a.as_ref().left().unwrap().iter(),
                zip(
                    self.b.as_ref().left().unwrap().iter(),
                    zip(
                        &self.scale_adapters,
                        self.dropout.as_ref().left().unwrap().iter(),
                    ),
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
                    .mul(*adapter_scale)?
                    .mul(global_scaling_weight)?;
                result = (result + res)?;
            }
            Ok(result)
        } else {
            let mut inputs = Vec::new();
            let a = self.a.as_ref().right().unwrap();
            let b = self.b.as_ref().right().unwrap();
            dbg!(a.weight());
            dbg!(b.weight());
            let dropout = self.dropout.as_ref().right().unwrap();
            let init = Tensor::zeros(input.shape(), input.dtype(), input.device())?.unsqueeze(0)?;
            for i in 0..self.n_adapters {
                let mut input_new = input.to_dtype(a.weight().dtype())?;
                input_new = apply_scalings_to_x(input_new.clone(), &scalings, i)?;

                input_new = if let Some(ref dropout) = dropout {
                    dropout.forward(&input_new, true)?
                } else {
                    input_new.clone()
                };
                inputs.push(input_new.unsqueeze(0)?);
            }
            let input = Tensor::cat(&inputs, 0)?;
            dbg!(&input);
            let out = b.forward(&a.forward(&input)?)?;
            dbg!(&out);
            let summed = (out
                .chunk(self.n_adapters, 0)?
                .iter()
                .fold(init, |acc, x| (acc + x).unwrap())
                * global_scaling_weight)?;
            dbg!(&summed);

            result + (summed.squeeze(0)?)
        }
    }
}
