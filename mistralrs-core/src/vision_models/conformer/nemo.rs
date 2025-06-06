use candle_core::Result;
use candle_nn::{Conv2dConfig, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers;

use super::config::NemoConvConfig;

pub struct NemoConvSubsampling {
    conv: Vec<Box<dyn Module>>,
    conv2d_subsampling: bool,
    out: Linear,
    subsampling_causal_cond: bool,
}

impl NemoConvSubsampling {
    pub fn new(cfg: &NemoConvConfig, vb: ShardedVarBuilder) -> Result<Self> {
        if cfg.subsampling_factor % 2 != 0 {
            candle_core::bail!("Sampling factor should be a multiple of 2!");
        }

        let sampling_num = (cfg.subsampling_factor as f32).log2() as usize;
        let subsampling_causal_cond =
            ["dw_striding", "striding", "striding_conv1d"].contains(&cfg.subsampling.as_str());

        let mut in_channels = 1;
        let mut layers: Vec<Box<dyn Module>> = Vec::new();

        let stride = 2;
        let kernel_size = 3;
        let ceil_mode = false;

        assert_eq!(cfg.is_causal, false);
        assert_eq!(cfg.subsampling, "dw_striding");

        let left_padding = (kernel_size - 1) / 2;
        let right_padding = (kernel_size - 1) / 2;
        let max_cache_len = 0;

        {
            let vb_layers = vb.pp("layers");

            let mut idx = 0;
            layers.push(Box::new(layers::conv2d(
                in_channels,
                cfg.conv_channels,
                kernel_size,
                Conv2dConfig {
                    padding: left_padding,
                    stride: stride,
                    dilation: 1,
                    groups: 1,
                },
                vb_layers.pp(idx),
            )?));

            in_channels = cfg.conv_channels;
            idx += 1;
            layers.push(Box::new(cfg.activation));

            for _ in 0..(sampling_num - 1) {
                idx += 1;
                layers.push(Box::new(layers::conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    Conv2dConfig {
                        padding: left_padding,
                        stride: stride,
                        dilation: 1,
                        groups: 1,
                    },
                    vb_layers.pp(idx),
                )?));
            }

            idx += 1;
            layers.push(Box::new(layers::conv2d(
                in_channels,
                in_channels,
                1,
                Conv2dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb_layers.pp(idx),
            )?));

            idx += 1;
            layers.push(Box::new(cfg.activation));
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
        let conv2d_subsampling = false;

        Ok(Self {
            conv: layers,
            conv2d_subsampling,
            out,
            subsampling_causal_cond,
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
        let add_pad = (all_paddings - kernel_size) as f32;
        let one = 1f32;
        for i in 0..repeat_num {
            length = (length + add_pad) / (stride as f32) + one;
            if ceil_mode {
                length = length.ceil();
            } else {
                length = length.floor();
            }
        }
        length as usize
    }
}
