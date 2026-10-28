use candle_core::{op::BackpropOp, CpuStorage, DType, Result, Storage, Tensor, WithDType};

const PROJECTION_PARTS: usize = 3;
const PARAM_COUNT: usize = 15;
const OPTIMIZED_KERNEL_WIDTH: usize = 7;

fn cpu_storage(storage: &Storage) -> Result<&CpuStorage> {
    match storage {
        Storage::Cpu(storage) => Ok(storage),
        _ => candle_core::bail!("LFM3 convolution expects CPU storage"),
    }
}

struct ConvInputs<'a> {
    projected: &'a Tensor,
    weight: &'a Tensor,
    bias: Option<&'a Tensor>,
    state: &'a Tensor,
    params: [u64; PARAM_COUNT],
    batch: usize,
    seq: usize,
    hidden: usize,
    width: usize,
}

impl ConvInputs<'_> {
    fn output_len(&self) -> usize {
        self.batch * self.seq * self.hidden
    }

    fn state_len(&self) -> usize {
        self.batch * self.hidden * self.width
    }

    fn tensors(&self, output: Storage, state: Storage) -> (Tensor, Tensor) {
        (
            Tensor::from_storage(
                output,
                (self.batch, self.seq, self.hidden),
                BackpropOp::none(),
                false,
            ),
            Tensor::from_storage(
                state,
                (self.batch, self.hidden, self.width),
                BackpropOp::none(),
                false,
            ),
        )
    }

    fn cpu(&self) -> Result<(Tensor, Tensor)> {
        let (projected, projected_layout) = self.projected.storage_and_layout();
        let (weight, weight_layout) = self.weight.storage_and_layout();
        let (state, state_layout) = self.state.storage_and_layout();
        let bias = self.bias.map(Tensor::storage_and_layout);
        let projected = cpu_storage(&projected)?;
        let weight = cpu_storage(&weight)?;
        let state = cpu_storage(&state)?;
        let bias = bias
            .as_ref()
            .map(|(storage, layout)| -> Result<_> { Ok((cpu_storage(storage)?, *layout)) })
            .transpose()?;

        macro_rules! run {
            ($ty:ty) => {{
                let input = CpuInputs {
                    projected: &projected.as_slice::<$ty>()?[projected_layout.start_offset()..],
                    weight: &weight.as_slice::<$ty>()?[weight_layout.start_offset()..],
                    state: &state.as_slice::<$ty>()?[state_layout.start_offset()..],
                    bias: bias
                        .map(|(storage, layout)| {
                            storage
                                .as_slice::<$ty>()
                                .map(|values| &values[layout.start_offset()..])
                        })
                        .transpose()?,
                    params: &self.params,
                };
                let (output, state) = if self.width == OPTIMIZED_KERNEL_WIDTH {
                    input.run::<OPTIMIZED_KERNEL_WIDTH>(self.output_len(), self.state_len())
                } else {
                    input.run::<0>(self.output_len(), self.state_len())
                };
                self.tensors(
                    Storage::Cpu(<$ty>::to_cpu_storage_owned(output)),
                    Storage::Cpu(<$ty>::to_cpu_storage_owned(state)),
                )
            }};
        }
        Ok(match self.projected.dtype() {
            DType::F32 => run!(f32),
            DType::F16 => run!(half::f16),
            DType::BF16 => run!(half::bf16),
            dtype => candle_core::bail!("Unsupported LFM3 convolution dtype {dtype:?}"),
        })
    }
}

struct CpuInputs<'a, T> {
    projected: &'a [T],
    weight: &'a [T],
    state: &'a [T],
    bias: Option<&'a [T]>,
    params: &'a [u64; PARAM_COUNT],
}

impl<T: WithDType> CpuInputs<'_, T> {
    #[inline]
    fn gated(&self, batch: usize, token: usize, channel: usize) -> T {
        let p = self.params;
        let hidden = p[2] as usize;
        let offset = batch * p[4] as usize + token * p[5] as usize + channel * p[6] as usize;
        T::from_f64(
            (self.projected[offset].to_f64() as f32
                * self.projected[offset + 2 * hidden * p[6] as usize].to_f64() as f32)
                as f64,
        )
    }

    #[inline]
    fn previous(&self, batch: usize, channel: usize, index: usize) -> T {
        let p = self.params;
        if p[13] == 0 {
            T::from_f64(0.0)
        } else {
            self.state[batch * p[7] as usize + channel * p[8] as usize + index * p[9] as usize]
        }
    }

    fn run<const WIDTH: usize>(&self, output_len: usize, state_len: usize) -> (Vec<T>, Vec<T>) {
        let p = self.params;
        let seq = p[1] as usize;
        let hidden = p[2] as usize;
        let width = if WIDTH == 0 { p[3] as usize } else { WIDTH };
        let mut output = vec![T::from_f64(0.0); output_len];
        let mut next_state = vec![T::from_f64(0.0); state_len];
        let output_ptr = output.as_mut_ptr() as usize;
        let state_ptr = next_state.as_mut_ptr() as usize;
        candle_core::utils::barrier_pool().execute_chunked(output_len + state_len, |range| {
            for index in range {
                if index < output_len {
                    let channel = index % hidden;
                    let token = index / hidden % seq;
                    let batch = index / (hidden * seq);
                    let mut sum = 0.0f32;
                    for tap in 0..width {
                        let value = if token + tap + 1 >= width {
                            self.gated(batch, token + tap + 1 - width, channel)
                        } else {
                            self.previous(batch, channel, token + tap + 1)
                        };
                        let weight = self.weight[channel * p[10] as usize + tap * p[11] as usize];
                        sum += value.to_f64() as f32 * weight.to_f64() as f32;
                    }
                    if let Some(bias) = self.bias {
                        sum += bias[channel * p[12] as usize].to_f64() as f32;
                    }
                    let convolved = T::from_f64(sum as f64);
                    let gate_offset = batch * p[4] as usize
                        + token * p[5] as usize
                        + (hidden + channel) * p[6] as usize;
                    let value = T::from_f64(
                        (convolved.to_f64() as f32 * self.projected[gate_offset].to_f64() as f32)
                            as f64,
                    );
                    // Each worker owns disjoint output indices.
                    unsafe { *((output_ptr as *mut T).add(index)) = value };
                } else {
                    let index = index - output_len;
                    let tap = index % width;
                    let channel = index / width % hidden;
                    let batch = index / (width * hidden);
                    let value = if seq + tap >= width {
                        self.gated(batch, seq + tap - width, channel)
                    } else {
                        self.previous(batch, channel, seq + tap)
                    };
                    unsafe { *((state_ptr as *mut T).add(index)) = value };
                }
            }
        });
        (output, next_state)
    }
}

pub fn lfm3_causal_conv1d(
    projected: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    state: &Tensor,
    use_existing_state: bool,
) -> Result<(Tensor, Tensor)> {
    let (batch, seq, projected_hidden) = projected.dims3()?;
    let (hidden, groups, width) = weight.dims3()?;
    if projected_hidden != PROJECTION_PARTS * hidden
        || groups != 1
        || width == 0
        || seq == 0
        || state.dims3()? != (batch, hidden, width)
        || bias.is_some_and(|bias| bias.dims() != [hidden])
    {
        candle_core::bail!("Incompatible LFM3 convolution input, weight, bias, or state shapes");
    }
    for tensor in [Some(weight), Some(state), bias].into_iter().flatten() {
        if tensor.dtype() != projected.dtype() || !tensor.device().same_device(projected.device()) {
            candle_core::bail!("LFM3 convolution tensors must share dtype and device");
        }
    }
    let x_stride = projected.stride();
    let state_stride = state.stride();
    let weight_stride = weight.stride();
    let inputs = ConvInputs {
        projected,
        weight,
        bias,
        state,
        params: [
            batch as u64,
            seq as u64,
            hidden as u64,
            width as u64,
            x_stride[0] as u64,
            x_stride[1] as u64,
            x_stride[2] as u64,
            state_stride[0] as u64,
            state_stride[1] as u64,
            state_stride[2] as u64,
            weight_stride[0] as u64,
            weight_stride[2] as u64,
            bias.map_or(0, |bias| bias.stride()[0] as u64),
            u64::from(use_existing_state),
            u64::from(bias.is_some()),
        ],
        batch,
        seq,
        hidden,
        width,
    };
    match projected.device() {
        candle_core::Device::Cpu => inputs.cpu(),
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_) => inputs.metal(),
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => inputs.cuda(),
        #[allow(unreachable_patterns)]
        _ => candle_core::bail!("LFM3 convolution backend is not enabled"),
    }
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "metal")]
mod metal;

#[cfg(test)]
mod tests;
