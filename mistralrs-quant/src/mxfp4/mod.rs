use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    AfqBits, AfqGroupSize, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedConfig, QuantizedSerde, ShardedVarBuilder,
};

use crate::afq::ops;

const GROUP_SIZE: AfqGroupSize = AfqGroupSize::Low;
const _: () = assert!(GROUP_SIZE as usize == 32);

const BITS: AfqBits = AfqBits::Mxfp4;
const _: () = assert!(BITS as usize == 40);

pub(crate) const N_BITS: usize = 4;

#[derive(Debug)]
pub struct MXFP4Layer {
    blocks: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
}

impl QuantMethod for MXFP4Layer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Afq { .. } => unreachable!(),
            QuantMethodConfig::MXFP4 {
                blocks,
                scales,
                bias,
            } => Ok(Self {
                blocks,
                scales,
                bias,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        ops::afq_dequantize_op(
            &self.blocks,
            &self.scales,
            &self.scales.clone(),
            GROUP_SIZE,
            BITS,
        )
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = ops::afq_mm_op(
            x,
            &self.blocks,
            &self.scales,
            &self.scales.clone(),
            None,
            None,
            GROUP_SIZE,
            BITS,
            true,
        )?;
        if let Some(bias) = &self.bias {
            x = x.broadcast_add(bias)?;
        }
        Ok(x)
    }

    fn gather_forward(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let mut x = ops::afq_mm_op(
            x,
            &self.blocks,
            &self.scales,
            &self.scales.clone(),
            None,
            Some(indices),
            GROUP_SIZE,
            BITS,
            true,
        )?;
        if let Some(bias) = &self.bias {
            x = x.broadcast_add(bias)?;
        }
        Ok(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("MXFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.scales.dtype(), self.scales.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }
}

impl MXFP4Layer {
    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !vb.device().is_metal() {
            candle_core::bail!("MXFP4Layer only works on Metal.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let group_size = GROUP_SIZE as usize;

        let blocks = vb.get_with_hints_dtype(
            (out_dim, in_dim * N_BITS / 32),
            "blocks",
            Default::default(),
            DType::F4,
        )?;
        let scales = vb.get_with_hints_dtype(
            (out_dim, in_dim / group_size),
            "scales",
            Default::default(),
            DType::F8E8M0,
        )?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    pub fn packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !vb.device().is_metal() {
            candle_core::bail!("MXFP4Layer only works on Metal.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let group_size = GROUP_SIZE as usize;

        let blocks = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim * N_BITS / 32),
            "blocks",
            Default::default(),
            DType::F4,
        )?;
        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / group_size),
            "scales",
            Default::default(),
            DType::F8E8M0,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }
}

impl QuantizedSerde for MXFP4Layer {
    fn name(&self) -> &'static str {
        "mxfp4-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        false
    }
}
