use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, ops::softmax_last_dim, Linear, Module, VarBuilder};

use super::config::XLoraConfig;

#[derive(Debug)]
struct TemperatureScaledSoftmax {
    temp: f64,
}

impl Module for TemperatureScaledSoftmax {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        softmax_last_dim(&(xs / self.temp)?)
    }
}

#[derive(Debug)]
pub struct XLoraClassifier {
    last: Linear,
    inner: Vec<Linear>,
    softmax: Option<TemperatureScaledSoftmax>,
    scaling_pass_value: f64,
    model_layers: usize,
    n_classes: usize,
    config: XLoraConfig,
}

impl XLoraClassifier {
    pub fn new(
        config: XLoraConfig,
        n_layers: usize,
        n_classes: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (last, inner) = if config.xlora_depth == 1 {
            if config.layerwise_scalings {
                assert!(vb.contains_tensor("last.weight"));
                (
                    linear(config.hidden_size, n_classes * n_layers, vb.pp("last")).unwrap(),
                    vec![],
                )
            } else {
                assert!(vb.contains_tensor("last.weight"));
                (
                    linear(config.hidden_size, n_classes, vb.pp("last"))?,
                    vec![],
                )
            }
        } else if config.xlora_depth == 2 {
            let mut inner = Vec::new();
            assert!(vb.contains_tensor("inner.0.weight"));
            inner.push(linear(
                config.hidden_size,
                config.xlora_size,
                vb.pp("inner.0"),
            )?);
            assert!(vb.contains_tensor("last.weight"));
            if config.layerwise_scalings {
                (
                    linear(config.xlora_size, n_classes * n_layers, vb.pp("last"))?,
                    inner,
                )
            } else {
                (linear(config.xlora_size, n_classes, vb.pp("last"))?, inner)
            }
        } else {
            let mut inner = Vec::new();
            assert!(vb.contains_tensor("inner.0.weight"));
            inner.push(linear(
                config.hidden_size,
                config.xlora_size,
                vb.pp("inner.0"),
            )?);
            for i in 1..=config.xlora_depth - 2 {
                assert!(vb.contains_tensor(&format!("inner.{i}.weight")));
                inner.push(linear(
                    config.xlora_size,
                    config.xlora_size,
                    vb.pp(&format!("inner.{i}")),
                )?)
            }
            assert!(vb.contains_tensor("last.weight"));
            if config.layerwise_scalings {
                (
                    linear(config.xlora_size, n_classes * n_layers, vb.pp("last"))?,
                    inner,
                )
            } else {
                (linear(config.xlora_size, n_classes, vb.pp("last"))?, inner)
            }
        };
        Ok(Self {
            last,
            inner,
            softmax: if config.enable_softmax {
                Some(TemperatureScaledSoftmax {
                    temp: config.softmax_temperature,
                })
            } else {
                None
            },
            scaling_pass_value: config.scaling_pass_value,
            model_layers: n_layers,
            n_classes,
            config,
        })
    }

    pub fn forward(&self, mut hidden_states: Tensor) -> Result<Tensor> {
        for layer in &self.inner {
            hidden_states = layer.forward(&hidden_states)?;
        }
        let mut logits = self.last.forward(&hidden_states)?;

        if !self.config.layerwise_scalings {
            logits = logits.unsqueeze(2)?;
            logits = logits.expand((
                logits.dims()[0],
                logits.dims()[1],
                self.model_layers,
                logits.dims()[3],
            ))?;
        }

        let mut scalings = logits.reshape((
            hidden_states.dims()[0],
            hidden_states.dims()[1],
            self.model_layers,
            self.n_classes,
        ))?;

        if self.config.enable_softmax {
            scalings = scalings.apply(self.softmax.as_ref().unwrap())?;
        }

        Ok(scalings)
    }

    pub fn get_dummy_scalings(
        &self,
        bs: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        Tensor::full(
            self.scaling_pass_value,
            (bs, seq_len, self.model_layers, self.n_classes),
            device,
        )?
        .to_dtype(dtype)
    }
}
