use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    activation, linear, linear_no_bias, ops::softmax_last_dim, Dropout, Linear, Module, ModuleT,
    VarBuilder,
};

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

pub struct XLoraClassifier {
    last: Linear,
    inner: Vec<Box<dyn ModuleT + Send + Sync>>,
    softmax: Option<TemperatureScaledSoftmax>,
    scaling_pass_value: f64,
    model_layers: usize,
    n_classes: usize,
    pub config: XLoraConfig,
}

impl XLoraClassifier {
    pub fn new(
        config: XLoraConfig,
        n_layers: usize,
        n_classes: usize,
        vb: VarBuilder,
        is_quantized: bool,
    ) -> Result<Self> {
        if config.top_k_lora.is_some() || config.enable_softmax_topk {
            // TODO(EricLBuehler): Implement
            candle_core::bail!("`top_k_lora` and `enable_softmax_topk` are not yet supported.");
        }
        let (last, inner): (Linear, Vec<Box<dyn ModuleT + Send + Sync>>) = if config.xlora_depth
            == 1
        {
            let dim = if config.layerwise_scalings {
                n_classes * n_layers
            } else {
                n_classes
            };
            assert!(vb.contains_tensor("last.weight"));
            if config.use_bias {
                assert!(vb.contains_tensor("last.bias"));
                let lin = linear(config.hidden_size, dim, vb.pp("last"))?;
                (
                    if is_quantized {
                        Linear::new(
                            lin.weight().to_dtype(DType::F32)?,
                            lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                        )
                    } else {
                        lin
                    },
                    vec![],
                )
            } else {
                let lin = linear_no_bias(config.hidden_size, dim, vb.pp("last"))?;
                (
                    if is_quantized {
                        Linear::new(
                            lin.weight().to_dtype(DType::F32)?,
                            lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                        )
                    } else {
                        lin
                    },
                    vec![],
                )
            }
        } else if config.xlora_depth == 2 {
            let mut inner: Vec<Box<dyn ModuleT + Send + Sync>> = Vec::new();
            assert!(vb.contains_tensor("inner.0.weight"));
            if config.use_bias {
                assert!(vb.contains_tensor("inner.0.bias"));
                let lin = linear(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?;
                inner.push(Box::new(if is_quantized {
                    Linear::new(
                        lin.weight().to_dtype(DType::F32)?,
                        lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                    )
                } else {
                    lin
                }));
            } else {
                let lin = linear_no_bias(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?;
                inner.push(Box::new(if is_quantized {
                    Linear::new(
                        lin.weight().to_dtype(DType::F32)?,
                        lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                    )
                } else {
                    lin
                }));
            }
            if config.enable_relu_and_dropout {
                inner.push(Box::new(activation::Activation::Relu));
                inner.push(Box::new(Dropout::new(config.xlora_dropout_p)));
            }
            let dim = if config.layerwise_scalings {
                n_classes * n_layers
            } else {
                n_classes
            };
            assert!(vb.contains_tensor("last.weight"));
            if config.use_bias {
                assert!(vb.contains_tensor("last.bias"));
                let lin = linear(config.hidden_size, dim, vb.pp("last"))?;
                (
                    if is_quantized {
                        Linear::new(
                            lin.weight().to_dtype(DType::F32)?,
                            lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                        )
                    } else {
                        lin
                    },
                    inner,
                )
            } else {
                let lin = linear_no_bias(config.hidden_size, dim, vb.pp("last"))?;
                (
                    if is_quantized {
                        Linear::new(
                            lin.weight().to_dtype(DType::F32)?,
                            lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                        )
                    } else {
                        lin
                    },
                    inner,
                )
            }
        } else {
            let mut inner: Vec<Box<dyn ModuleT + Send + Sync>> = Vec::new();
            assert!(vb.contains_tensor("inner.0.weight"));
            if config.use_bias {
                assert!(vb.contains_tensor("inner.0.bias"));
                let lin = linear(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?;
                inner.push(Box::new(if is_quantized {
                    Linear::new(
                        lin.weight().to_dtype(DType::F32)?,
                        lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                    )
                } else {
                    lin
                }));
            } else {
                let lin = linear_no_bias(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?;
                inner.push(Box::new(if is_quantized {
                    Linear::new(
                        lin.weight().to_dtype(DType::F32)?,
                        lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                    )
                } else {
                    lin
                }));
            }
            if config.enable_relu_and_dropout {
                inner.push(Box::new(activation::Activation::Relu));
                inner.push(Box::new(Dropout::new(config.xlora_dropout_p)));
            }
            for i in 1..=config.xlora_depth - 2 {
                assert!(vb.contains_tensor(&format!("inner.{i}.weight")));
                if config.use_bias {
                    assert!(vb.contains_tensor(&format!("inner.{i}.bias")));
                    let lin = linear(
                        config.xlora_size,
                        config.xlora_size,
                        vb.pp(&format!("inner.{i}")),
                    )?;
                    inner.push(Box::new(Linear::new(
                        lin.weight().to_dtype(DType::F32)?,
                        lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                    )));
                } else {
                    let lin = linear_no_bias(
                        config.xlora_size,
                        config.xlora_size,
                        vb.pp(&format!("inner.{i}")),
                    )?;
                    inner.push(Box::new(Linear::new(
                        lin.weight().to_dtype(DType::F32)?,
                        lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                    )));
                }
                if config.enable_relu_and_dropout {
                    inner.push(Box::new(activation::Activation::Relu));
                    inner.push(Box::new(Dropout::new(config.xlora_dropout_p)));
                }
            }
            let dim = if config.layerwise_scalings {
                n_classes * n_layers
            } else {
                n_classes
            };
            assert!(vb.contains_tensor("last.weight"));
            if config.use_bias {
                assert!(vb.contains_tensor("last.bias"));
                let lin = linear(config.hidden_size, dim, vb.pp("last"))?;
                (
                    if is_quantized {
                        Linear::new(
                            lin.weight().to_dtype(DType::F32)?,
                            lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                        )
                    } else {
                        lin
                    },
                    inner,
                )
            } else {
                let lin = linear_no_bias(config.hidden_size, dim, vb.pp("last"))?;
                (
                    if is_quantized {
                        Linear::new(
                            lin.weight().to_dtype(DType::F32)?,
                            lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                        )
                    } else {
                        lin
                    },
                    inner,
                )
            }
        };
        let last = if is_quantized {
            Linear::new(
                last.weight().to_dtype(DType::F32)?,
                last.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            last
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
            hidden_states = layer.forward_t(&hidden_states, true)?;
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
            logits.dims()[0],
            logits.dims()[1],
            self.model_layers,
            self.n_classes,
        ))?;
        if let Some(ref softmax) = self.softmax {
            scalings = softmax.forward(&scalings)?;
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

    pub fn get_global_scaling_weight(&self) -> f64 {
        self.config.global_scaling_weight
    }
}
