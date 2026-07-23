use crate::speech_models::pockettts::voice_state::ModelState;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use std::collections::HashMap;

#[derive(Clone)]
pub struct StreamingConv1d {
    conv: Conv1d,
    padding_mode: String,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    in_channels: usize,
    name: String,
}

impl StreamingConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        padding_mode: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            stride,
            padding: 0,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if bias {
            candle_nn::conv1d(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("conv"),
            )?
        } else {
            candle_nn::conv1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("conv"),
            )?
        };

        Ok(Self {
            conv,
            padding_mode: padding_mode.to_string(),
            stride,
            kernel_size,
            dilation,
            in_channels,
            name: name.to_string(),
        })
    }

    pub fn effective_kernel_size(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let kernel = self.effective_kernel_size();
        let mut state = HashMap::new();
        if kernel > self.stride {
            let previous = Tensor::zeros(
                (batch_size, self.in_channels, kernel - self.stride),
                DType::F32,
                device,
            )?;
            state.insert("previous".to_string(), previous);
        }
        Ok(state)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState, step: usize) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let s = self.stride;
        if t == 0 || t % s != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Steps must be multiple of stride {}, got {}",
                s, t
            )));
        }

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            let init = self.init_state(b, t, x.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();
        let kernel = self.effective_kernel_size();
        let pad_left = kernel.saturating_sub(s);

        if pad_left > 0 {
            let previous = module_state
                .remove("previous")
                .ok_or_else(|| candle_core::Error::Msg("previous state not found".to_string()))?;
            let is_first = step == 0;

            let x_with_padding = if is_first && self.padding_mode == "replicate" {
                // Replicate the first frame for the initial padding
                let first_frame = x.narrow(2, 0, 1)?;
                let replicated_padding = first_frame.broadcast_as((b, c, pad_left))?;
                Tensor::cat(&[replicated_padding, x.clone()], 2)?
            } else {
                Tensor::cat(&[previous, x.clone()], 2)?
            };

            let y = self.conv.forward(&x_with_padding)?;

            // Update previous state for next call
            let total_len = x_with_padding.dims()[2];
            let new_previous = x_with_padding.narrow(2, total_len - pad_left, pad_left)?;
            module_state.insert("previous".to_string(), new_previous);

            Ok(y)
        } else {
            self.conv.forward(x)
        }
    }

    pub fn weight(&self) -> &Tensor {
        self.conv.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.conv.bias()
    }
}

#[derive(Clone)]
pub struct StreamingConvTranspose1d {
    convtr: ConvTranspose1d,
    stride: usize,
    kernel_size: usize,
    out_channels: usize,
    name: String,
}

impl StreamingConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            stride,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups,
        };
        let convtr = if bias {
            candle_nn::conv_transpose1d(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("convtr"),
            )?
        } else {
            candle_nn::conv_transpose1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("convtr"),
            )?
        };

        Ok(Self {
            convtr,
            stride,
            kernel_size,
            out_channels,
            name: name.to_string(),
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        let k = self.kernel_size;
        let s = self.stride;
        if k > s {
            let partial =
                Tensor::zeros((batch_size, self.out_channels, k - s), DType::F32, device)?;
            state.insert("partial".to_string(), partial);
        }
        Ok(state)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        _step: usize,
    ) -> Result<Tensor> {
        let (b, _c, t) = x.dims3()?;
        let k = self.kernel_size;
        let s = self.stride;
        let trim = k.saturating_sub(s);

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            let init = self.init_state(b, t, x.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();

        let mut y = self.convtr.forward(x)?;

        if trim > 0 {
            if let Some(partial) = module_state.remove("partial") {
                // y is (B, C, S*T + trim)
                // We add partial to the start of y
                let y_head = y.narrow(2, 0, trim)?;
                let y_sum = (y_head + partial)?;
                let y_tail = y.narrow(2, trim, y.dims()[2] - trim)?;
                y = Tensor::cat(&[y_sum, y_tail], 2)?;
            }

            // The last `trim` elements of `y` become the next `partial`
            let len = y.dims()[2];
            let mut next_partial = y.narrow(2, len - trim, trim)?;

            // If bias exists, we need to subtract it from the partial state
            // because it will be added again when we run the next forward pass.
            if let Some(bias) = self.convtr.bias() {
                let b_reshaped = bias.reshape((self.out_channels, 1))?;
                next_partial = next_partial.broadcast_sub(&b_reshaped)?;
            }
            module_state.insert("partial".to_string(), next_partial);

            // The output we actually return is y MINUS the new partial tail
            y = y.narrow(2, 0, len - trim)?;
        }

        Ok(y)
    }

    pub fn weight(&self) -> &Tensor {
        self.convtr.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.convtr.bias()
    }
}

#[derive(Clone)]
pub struct ConvDownsample1d {
    conv: StreamingConv1d,
}

impl ConvDownsample1d {
    pub fn new(stride: usize, dimension: usize, name: &str, vb: VarBuilder) -> Result<Self> {
        let conv = StreamingConv1d::new(
            dimension,
            dimension,
            2 * stride,
            stride,
            1,
            1,
            false,
            "replicate",
            &format!("{}.conv", name),
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        self.conv.init_state(batch_size, sequence_length, device)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState, step: usize) -> Result<Tensor> {
        self.conv.forward(x, model_state, step)
    }
}

#[derive(Clone)]
pub struct ConvTrUpsample1d {
    convtr: StreamingConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn new(stride: usize, dimension: usize, name: &str, vb: VarBuilder) -> Result<Self> {
        let convtr = StreamingConvTranspose1d::new(
            dimension,
            dimension,
            2 * stride,
            stride,
            dimension,
            false,
            &format!("{}.convtr", name),
            vb.pp("convtr"),
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        self.convtr.init_state(batch_size, sequence_length, device)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState, step: usize) -> Result<Tensor> {
        self.convtr.forward(x, model_state, step)
    }
}
