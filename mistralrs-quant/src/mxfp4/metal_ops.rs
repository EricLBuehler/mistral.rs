use candle_core::{backend::BackendStorage, MetalStorage, Result, Shape, Storage, Tensor};

use super::MXFP4_BLOCK_SIZE;

pub fn mxfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let scale = scale.contiguous()?;
    let bias = match bias {
        Some(b) => Some(b.contiguous()?),
        None => None,
    };

    let input_dims = input.dims();
    let weight_dims = weight.dims();
    let scale_dims = scale.dims();

    if input_dims.len() != 2 {
        candle_core::bail!("Expected input to be rank 2, got {:?}", input_dims);
    }
    if weight_dims.len() != 2 || scale_dims.len() != 2 {
        candle_core::bail!(
            "Expected weight and scale to be rank 2, got {:?} and {:?}",
            weight_dims,
            scale_dims
        );
    }

    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight_dims[0];

    if k % MXFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("MXFP4 requires K divisible by {MXFP4_BLOCK_SIZE}, got K={k}");
    }
    if k % 2 != 0 {
        candle_core::bail!("MXFP4 requires K divisible by 2, got K={k}");
    }
    if weight_dims[1] != k / 2 {
        candle_core::bail!(
            "Weight shape mismatch: expected [N, K/2] = [{}, {}], got {:?}",
            n,
            k / 2,
            weight_dims
        );
    }
    if scale_dims[0] != n || scale_dims[1] != k / MXFP4_BLOCK_SIZE {
        candle_core::bail!(
            "Scale shape mismatch: expected [N, K/32] = [{}, {}], got {:?}",
            n,
            k / MXFP4_BLOCK_SIZE,
            scale_dims
        );
    }

    let input_s = input.storage_and_layout().0;
    let weight_s = weight.storage_and_layout().0;
    let scale_s = scale.storage_and_layout().0;

    let Storage::Metal(input_s) = &*input_s else {
        candle_core::bail!("Expected Metal storage for input")
    };
    let Storage::Metal(weight_s) = &*weight_s else {
        candle_core::bail!("Expected Metal storage for weight")
    };
    let Storage::Metal(scale_s) = &*scale_s else {
        candle_core::bail!("Expected Metal storage for scale")
    };

    let device = input_s.device();
    let encoder = device.command_encoder()?;
    encoder.set_label("mxfp4-matmul");

    let output = device.new_buffer(m * n, input.dtype(), "mxfp4-matmul-output")?;

    let x = (
        input_s.buffer(),
        input.layout().start_offset() * input_s.dtype().size_in_bytes(),
    );
    let w = (
        weight_s.buffer(),
        weight.layout().start_offset() * weight_s.dtype().size_in_bytes(),
    );
    let scales = (
        scale_s.buffer(),
        scale.layout().start_offset() * scale_s.dtype().size_in_bytes(),
    );

    if let Some(bias) = &bias {
        let bias_s = bias.storage_and_layout().0;
        let Storage::Metal(bias_s) = &*bias_s else {
            candle_core::bail!("Expected Metal storage for bias")
        };
        if bias.dtype() != input.dtype() {
            candle_core::bail!(
                "Bias dtype mismatch: input={:?}, bias={:?}",
                input.dtype(),
                bias.dtype()
            );
        }
        if bias.dims() != [n] {
            candle_core::bail!(
                "Bias shape mismatch: expected [N]=[{n}], got {:?}",
                bias.dims()
            );
        }
        let bias = (
            bias_s.buffer(),
            bias.layout().start_offset() * bias_s.dtype().size_in_bytes(),
        );

        crate::metal_kernels::call_mxfp4_matmul(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            input.dtype(),
            x,
            w,
            scales,
            bias,
            &output,
            m,
            n,
            k,
            true,
        )
        .map_err(candle_core::Error::wrap)?;
    } else {
        // Any valid buffer is fine as long as has_bias=false.
        let dummy_bias = (input_s.buffer(), 0usize);

        crate::metal_kernels::call_mxfp4_matmul(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            input.dtype(),
            x,
            w,
            scales,
            dummy_bias,
            &output,
            m,
            n,
            k,
            false,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            m * n,
            input.dtype(),
        )),
        Shape::from((m, n)),
    )))
}

pub fn mxfp4_indexed_moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    biases: Option<&Tensor>,
    indices: &Tensor,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let weight_scales = weight_scales.contiguous()?;
    let indices = indices.contiguous()?;
    let biases = match biases {
        Some(b) => Some(b.contiguous()?),
        None => None,
    };

    let indices_dims = indices.dims();
    if indices_dims.len() != 2 {
        candle_core::bail!(
            "Expected indices to be rank 2 [num_tokens, topk], got {:?}",
            indices_dims
        );
    }
    let num_tokens = indices_dims[0];
    let topk = indices_dims[1];

    let input_dims = input.dims();
    let (k, input_has_topk_dim) = match input_dims {
        [t, kk] => {
            if *t != num_tokens {
                candle_core::bail!(
                    "Input/indices mismatch: input num_tokens={t}, indices num_tokens={num_tokens}"
                );
            }
            (*kk, false)
        }
        [t, tk, kk] => {
            if *t != num_tokens || *tk != topk {
                candle_core::bail!(
                    "Input/indices mismatch: input dims={input_dims:?}, indices dims={indices_dims:?}"
                );
            }
            (*kk, true)
        }
        _ => candle_core::bail!("Expected input to be rank 2 or 3, got {:?}", input_dims),
    };

    if k % MXFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("MXFP4 requires K divisible by {MXFP4_BLOCK_SIZE}, got K={k}");
    }
    if k % 2 != 0 {
        candle_core::bail!("MXFP4 requires K divisible by 2, got K={k}");
    }

    let w_dims = weights.dims();
    let s_dims = weight_scales.dims();
    if w_dims.len() != 3 || s_dims.len() != 3 {
        candle_core::bail!(
            "Expected weights and scales to be rank 3 [E, N, *], got {:?} and {:?}",
            w_dims,
            s_dims
        );
    }
    let num_experts = w_dims[0];
    let n = w_dims[1];

    if w_dims[2] != k / 2 {
        candle_core::bail!(
            "Weights shape mismatch: expected [E, N, K/2] = [{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / 2,
            w_dims
        );
    }
    if s_dims[0] != num_experts || s_dims[1] != n || s_dims[2] != k / MXFP4_BLOCK_SIZE {
        candle_core::bail!(
            "Scales shape mismatch: expected [E, N, K/32] = [{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / MXFP4_BLOCK_SIZE,
            s_dims
        );
    }

    let input_s = input.storage_and_layout().0;
    let weights_s = weights.storage_and_layout().0;
    let scales_s = weight_scales.storage_and_layout().0;
    let indices_s = indices.storage_and_layout().0;

    let Storage::Metal(input_s) = &*input_s else {
        candle_core::bail!("Expected Metal storage for input")
    };
    let Storage::Metal(weights_s) = &*weights_s else {
        candle_core::bail!("Expected Metal storage for weights")
    };
    let Storage::Metal(scales_s) = &*scales_s else {
        candle_core::bail!("Expected Metal storage for weight scales")
    };
    let Storage::Metal(indices_s) = &*indices_s else {
        candle_core::bail!("Expected Metal storage for indices")
    };

    let device = input_s.device();
    let encoder = device.command_encoder()?;
    encoder.set_label("mxfp4-indexed-moe-gemm");

    let output = device.new_buffer(
        num_tokens * topk * n,
        input.dtype(),
        "mxfp4-indexed-moe-gemm-output",
    )?;

    // If the input doesn't have a topk dim, we can reuse the loaded input across all experts
    // when topk is small. Otherwise, dispatch one threadgroup per expert_slot.
    let reuse_topk = !input_has_topk_dim && topk <= 8;

    let x = (
        input_s.buffer(),
        input.layout().start_offset() * input_s.dtype().size_in_bytes(),
    );
    let w = (
        weights_s.buffer(),
        weights.layout().start_offset() * weights_s.dtype().size_in_bytes(),
    );
    let scales = (
        scales_s.buffer(),
        weight_scales.layout().start_offset() * scales_s.dtype().size_in_bytes(),
    );
    let indices = (
        indices_s.buffer(),
        indices.layout().start_offset() * indices_s.dtype().size_in_bytes(),
    );

    if let Some(biases) = &biases {
        if biases.dtype() != input.dtype() {
            candle_core::bail!(
                "Bias dtype mismatch: input={:?}, bias={:?}",
                input.dtype(),
                biases.dtype()
            );
        }
        if biases.dims() != [num_experts, n] {
            candle_core::bail!(
                "Bias shape mismatch: expected [E, N]=[{num_experts}, {n}], got {:?}",
                biases.dims()
            );
        }

        let b_s = biases.storage_and_layout().0;
        let Storage::Metal(b_s) = &*b_s else {
            candle_core::bail!("Expected Metal storage for bias")
        };
        let bias = (
            b_s.buffer(),
            biases.layout().start_offset() * b_s.dtype().size_in_bytes(),
        );

        crate::metal_kernels::call_mxfp4_moe_gemm(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            input.dtype(),
            x,
            w,
            scales,
            bias,
            indices,
            &output,
            num_tokens,
            topk,
            num_experts,
            n,
            k,
            true,
            input_has_topk_dim,
            reuse_topk,
        )
        .map_err(candle_core::Error::wrap)?;
    } else {
        // Any valid buffer is fine as long as has_bias=false.
        let dummy_biases = (input_s.buffer(), 0usize);

        crate::metal_kernels::call_mxfp4_moe_gemm(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            input.dtype(),
            x,
            w,
            scales,
            dummy_biases,
            indices,
            &output,
            num_tokens,
            topk,
            num_experts,
            n,
            k,
            false,
            input_has_topk_dim,
            reuse_topk,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            num_tokens * topk * n,
            input.dtype(),
        )),
        Shape::from((num_tokens, topk, n)),
    )))
}
