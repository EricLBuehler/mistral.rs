#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers;

use super::config::NemoConvConfig;

pub struct NemoConvSubsampling {
    conv: Vec<Arc<dyn Module + Send + Sync>>,
    out: Linear,
    subsampling_factor: usize,
}

impl NemoConvSubsampling {
    pub fn new(cfg: &NemoConvConfig, vb: ShardedVarBuilder) -> Result<Self> {
        if !cfg.subsampling_factor.is_multiple_of(2) {
            candle_core::bail!("Sampling factor should be a multiple of 2!");
        }

        let sampling_num = (cfg.subsampling_factor as f32).log2() as usize;

        let mut in_channels = 1;
        let mut layers: Vec<Arc<dyn Module + Send + Sync>> = Vec::new();

        let stride = 2;
        let kernel_size = 3;
        let ceil_mode = false;

        assert!(!cfg.is_causal);
        assert_eq!(cfg.subsampling, "dw_striding");

        let left_padding = (kernel_size - 1) / 2;
        let right_padding = (kernel_size - 1) / 2;
        // let max_cache_len = if cfg.is_causal {
        //     cfg.subsampling_factor+1
        // } else {
        //     0
        // };

        {
            let vb_layers = vb.pp("conv");

            let mut idx = 0;
            layers.push(Arc::new(layers::conv2d(
                in_channels,
                cfg.conv_channels,
                kernel_size,
                Conv2dConfig {
                    padding: left_padding,
                    stride,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb_layers.pp(idx),
            )?));

            in_channels = cfg.conv_channels;
            idx += 1;
            layers.push(Arc::new(cfg.activation));

            for _ in 0..(sampling_num - 1) {
                idx += 1;
                layers.push(Arc::new(layers::conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    Conv2dConfig {
                        padding: left_padding,
                        stride,
                        dilation: 1,
                        groups: in_channels,
                        cudnn_fwd_algo: None,
                    },
                    vb_layers.pp(idx),
                )?));

                idx += 1;
                layers.push(Arc::new(layers::conv2d(
                    in_channels,
                    cfg.conv_channels,
                    1,
                    Conv2dConfig {
                        padding: 0,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                    vb_layers.pp(idx),
                )?));

                idx += 1;
                layers.push(Arc::new(cfg.activation));
            }
        }

        let in_length = cfg.feat_in as f32;
        let out_length = Self::calc_length(
            in_length,
            left_padding + right_padding,
            kernel_size,
            stride,
            ceil_mode,
            sampling_num,
        );
        let out = layers::linear_b(
            cfg.conv_channels * out_length,
            cfg.feat_out,
            true,
            vb.pp("out"),
        )?;

        Ok(Self {
            conv: layers,
            out,
            subsampling_factor: cfg.subsampling_factor,
        })
    }

    fn calc_length(
        mut length: f32,
        all_paddings: usize,
        kernel_size: usize,
        stride: usize,
        ceil_mode: bool,
        repeat_num: usize,
    ) -> usize {
        let add_pad = all_paddings as f32 - kernel_size as f32;
        let one = 1f32;
        for _ in 0..repeat_num {
            length = (length + add_pad) / (stride as f32) + one;
            if ceil_mode {
                length = length.ceil();
            } else {
                length = length.floor();
            }
        }
        length as usize
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        // Unsqueeze channel axis for 2D conv
        let mut x = x.unsqueeze(1)?;

        // Apply conv layers
        // Dont do self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling
        for layer in &self.conv {
            x = layer.forward(&x)?;
        }

        // Flatten and apply output linear layer
        let (b, c, t, f) = x.dims4()?;
        x = x.transpose(1, 2)?.reshape((b, t, c * f))?;
        x = x.apply(&self.out)?;

        // Handle mask
        let new_mask = if let Some(mask) = mask {
            let max_audio_length = x.dim(1)?;
            let feature_lens = mask.sum_keepdim(1)?;
            let padding_length = feature_lens.apply(&|t: &Tensor| {
                (t.to_dtype(candle_core::DType::F32)? / self.subsampling_factor as f64)?.ceil()
            })?;

            let device = x.device();
            let arange = Tensor::arange(0u32, max_audio_length as u32, device)?
                .unsqueeze(0)?
                .broadcast_as((padding_length.dim(0)?, max_audio_length))?;
            let pad_mask = arange.lt(&padding_length)?;
            Some(pad_mask.unsqueeze(1)?)
        } else {
            None
        };

        Ok((x, new_mask))
    }
}
