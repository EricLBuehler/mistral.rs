use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Tensor};

use crate::{
    utils::{
        deserialize_tensor, fake_deserialize_tensor, serialize_tensor, version_is_compatible,
        UQFF_VERSION,
    },
    Comm, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    QuantizedSerde, QuantizedSerdeType, ShardedVarBuilder,
};

pub(crate) mod ops;

#[cfg(feature = "cuda")]
pub(crate) mod ffi;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum AfqBits {
    Two = 2,
    Three = 3,
    Four = 4,
    Six = 6,
    Eight = 8,
    Mxfp4 = 40,
}

impl TryFrom<usize> for AfqBits {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> Result<Self> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            6 => Ok(Self::Six),
            8 => Ok(Self::Eight),
            40 => Ok(Self::Mxfp4),
            x => candle_core::bail!("Invalid AFQ bits {x}."),
        }
    }
}

impl TryFrom<u8> for AfqBits {
    type Error = candle_core::Error;
    fn try_from(value: u8) -> Result<Self> {
        Self::try_from(value as usize)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Default)]
pub enum AfqGroupSize {
    Low = 32,
    #[default]
    Med = 64,
    High = 128,
}

impl TryFrom<usize> for AfqGroupSize {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> Result<Self> {
        match value {
            32 => Ok(Self::Low),
            64 => Ok(Self::Med),
            128 => Ok(Self::High),
            x => candle_core::bail!("Invalid AFQ group size {x}."),
        }
    }
}

impl TryFrom<u8> for AfqGroupSize {
    type Error = candle_core::Error;
    fn try_from(value: u8) -> Result<Self> {
        Self::try_from(value as usize)
    }
}

#[derive(Debug)]
pub struct AfqLayer {
    w_q: Tensor,
    scales: Tensor,
    biases: Tensor,
    bias: Option<Tensor>,
    bits: AfqBits,
    group_size: AfqGroupSize,
}

impl QuantMethod for AfqLayer {
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
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
            QuantMethodConfig::Afq {
                weight,
                bias,
                bits,
                group_size,
            } => {
                let (w_q, scales, biases) = ops::afq_quantize_op(&weight, group_size, bits)?;

                Ok(Self {
                    w_q,
                    scales,
                    biases,
                    bias,
                    bits,
                    group_size,
                })
            }
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        ops::afq_dequantize_op(
            &self.w_q,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::afq_mm_op(
            x,
            &self.w_q,
            &self.scales,
            &self.biases,
            None,
            None,
            self.group_size,
            self.bits,
            true,
        )
    }

    fn gather_forward(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        ops::afq_mm_op(
            x,
            &self.w_q,
            &self.scales,
            &self.biases,
            None,
            Some(indices),
            self.group_size,
            self.bits,
            true,
        )
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize_w()?;
        Ok(Arc::new(Self::new(QuantMethodConfig::Afq {
            weight: (dequant + delta)?,
            bias: self.bias.clone(),
            bits: self.bits,
            group_size: self.group_size,
        })?))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.scales.dtype(), self.scales.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                let w = self.dequantize_w()?.to_device(&device)?;
                let b = self
                    .bias
                    .as_ref()
                    .map(|b| b.to_device(&device))
                    .transpose()?;
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            _ => todo!(),
        }
    }
}

impl AfqLayer {
    pub fn get_isq_type_from_uqff(data: Cow<[u8]>) -> Result<IsqType> {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Afq as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Afq as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        // Weight, scales, biases
        fake_deserialize_tensor(&mut buffer)?;
        fake_deserialize_tensor(&mut buffer)?;
        fake_deserialize_tensor(&mut buffer)?;

        // Bits and group size
        let bits: AfqBits = buffer.read_u8()?.try_into()?;
        let _group_size: AfqGroupSize = buffer.read_u8()?.try_into()?;

        if has_bias {
            fake_deserialize_tensor(&mut buffer)?
        }

        match bits {
            AfqBits::Two => Ok(IsqType::AFQ2),
            AfqBits::Three => Ok(IsqType::AFQ3),
            AfqBits::Four => Ok(IsqType::AFQ4),
            AfqBits::Six => Ok(IsqType::AFQ6),
            AfqBits::Eight => Ok(IsqType::AFQ8),
            AfqBits::Mxfp4 => candle_core::bail!("mxfp4 is not supported as an ISQ type"),
        }
    }

    pub fn afq_linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let QuantizedConfig::Afq { bits, group_size } = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let w_q = vb.get_with_hints_dtype(
            (out_dim, in_dim * bits / 32),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let scales =
            vb.get_with_hints((out_dim, in_dim / group_size), "scales", Default::default())?;
        let biases =
            vb.get_with_hints((out_dim, in_dim / group_size), "biases", Default::default())?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            w_q,
            scales,
            bias,
            biases,
            bits: AfqBits::try_from(*bits)?,
            group_size: AfqGroupSize::try_from(*group_size)?,
        }))
    }

    pub fn afq_packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let QuantizedConfig::Afq { bits, group_size } = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let w_q = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim * bits / 32),
            "weight",
            Default::default(),
            DType::U32,
        )?;
        let scales = vb.get_with_hints(
            (num_local_experts, out_dim, in_dim / group_size),
            "scales",
            Default::default(),
        )?;
        let biases = vb.get_with_hints(
            (num_local_experts, out_dim, in_dim / group_size),
            "biases",
            Default::default(),
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            w_q,
            scales,
            bias,
            biases,
            bits: AfqBits::try_from(*bits)?,
            group_size: AfqGroupSize::try_from(*group_size)?,
        }))
    }
}

impl QuantizedSerde for AfqLayer {
    fn name(&self) -> &'static str {
        "afq-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        // Version is always first!
        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type for afq is 4
        buffer.push(QuantizedSerdeType::Afq as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Weight, scales, biases
        serialize_tensor(&mut buffer, &self.w_q)?;
        serialize_tensor(&mut buffer, &self.scales)?;
        serialize_tensor(&mut buffer, &self.biases)?;

        // Bits and group size
        buffer.push(self.bits as u8);
        buffer.push(self.group_size as u8);

        if let Some(bias) = &bias {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }
    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Afq as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Afq as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        // Weight, scales, biases
        let w_q = deserialize_tensor(&mut buffer, device)?;
        let scales = deserialize_tensor(&mut buffer, device)?;
        let biases = deserialize_tensor(&mut buffer, device)?;

        // Bits and group size
        let bits: AfqBits = buffer.read_u8()?.try_into()?;
        let group_size: AfqGroupSize = buffer.read_u8()?.try_into()?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            w_q,
            scales,
            bias: b,
            biases,
            bits,
            group_size,
        }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Afq as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Afq as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        // Weight, scales, biases
        let w_q = deserialize_tensor(&mut buffer, device)?;
        let scales = deserialize_tensor(&mut buffer, device)?;
        let biases = deserialize_tensor(&mut buffer, device)?;

        // Bits and group size
        let bits: AfqBits = buffer.read_u8()?.try_into()?;
        let group_size: AfqGroupSize = buffer.read_u8()?.try_into()?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok((
            Arc::new(Self {
                w_q,
                scales,
                bias: None,
                biases,
                bits,
                group_size,
            }),
            b,
        ))
    }
}
