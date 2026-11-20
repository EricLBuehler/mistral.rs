use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantizedSerdeType {
    Gguf = 0,
    Unquant = 1,
    Hqq = 2,
    Fp8 = 3,
    Afq = 4,
    F8Q8 = 5,
    Mxfp4 = 6,
}

impl QuantizedSerdeType {
    pub const ALL: [Self; 7] = [
        Self::Gguf,
        Self::Unquant,
        Self::Hqq,
        Self::Fp8,
        Self::Afq,
        Self::F8Q8,
        Self::Mxfp4,
    ];

    pub fn stored_label(self, quant_group: &str) -> String {
        match self {
            Self::Gguf => {
                if quant_group.starts_with('q') {
                    quant_group.to_string()
                } else {
                    "gguf".to_string()
                }
            }
            Self::Unquant => "unquant".to_string(),
            Self::Hqq => {
                if quant_group.starts_with("hqq") {
                    quant_group.to_string()
                } else {
                    "hqq".to_string()
                }
            }
            Self::Fp8 => "fp8".to_string(),
            Self::Afq => {
                if quant_group.starts_with("afq") {
                    quant_group.to_string()
                } else {
                    "afq".to_string()
                }
            }
            Self::F8Q8 => "f8q8".to_string(),
            Self::Mxfp4 => "mxfp4".to_string(),
        }
    }

    pub(crate) fn inspect_uqff_header(
        self,
        layer: &uqff::UqffLayerHeaderView<'_>,
    ) -> Option<uqff::UqffHeaderMatch> {
        match self {
            Self::Gguf => GgufMatMul::inspect_uqff_header(layer),
            Self::Unquant => UnquantLinear::inspect_uqff_header(layer),
            Self::Hqq => HqqLayer::inspect_uqff_header(layer),
            Self::Fp8 => FP8Linear::inspect_uqff_header(layer),
            Self::Afq => AfqLayer::inspect_uqff_header(layer),
            Self::F8Q8 => F8Q8Linear::inspect_uqff_header(layer),
            Self::Mxfp4 => MXFP4Layer::inspect_uqff_header(layer),
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        self,
        tensors: &[uqff::UqffTensor],
        prefix: &str,
    ) -> Result<String> {
        match self {
            Self::Gguf => GgufMatMul::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Unquant => UnquantLinear::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Hqq => HqqLayer::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Fp8 => FP8Linear::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Afq => AfqLayer::stored_label_from_uqff_tensors(tensors, prefix),
            Self::F8Q8 => F8Q8Linear::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Mxfp4 => MXFP4Layer::stored_label_from_uqff_tensors(tensors, prefix),
        }
    }
}

impl TryFrom<usize> for QuantizedSerdeType {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gguf),
            1 => Ok(Self::Unquant),
            2 => Ok(Self::Hqq),
            3 => Ok(Self::Fp8),
            4 => Ok(Self::Afq),
            5 => Ok(Self::F8Q8),
            6 => Ok(Self::Mxfp4),
            other => candle_core::bail!("QuantizedSerdeType {other} is invalid."),
        }
    }
}

pub trait QuantizedSerde {
    fn name(&self) -> &'static str;
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn uqff_type(&self) -> Option<IsqType> {
        None
    }
    fn serialize_uqff(&self, _prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        candle_core::bail!(
            "`{}` does not support UQFF serialization for {ty}.",
            self.name()
        )
    }
    fn deserialize_uqff(
        _reader: &UqffReader,
        _prefix: &str,
        _device: &Device,
        _shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!(
            "`{}` does not support UQFF deserialization.",
            std::any::type_name::<Self>()
        )
    }
    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType>
    where
        Self: Sized,
    {
        candle_core::bail!(
            "`{}` does not support UQFF type detection.",
            std::any::type_name::<Self>()
        )
    }
}
